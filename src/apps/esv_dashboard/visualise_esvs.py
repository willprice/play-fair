import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
import socket
from urllib.parse import parse_qsl, urlencode, urlparse

import flask
from cached_property import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import pandas as pd
import numpy as np

import dash
from dash import Dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from dash.exceptions import PreventUpdate
from flask import make_response

parser = argparse.ArgumentParser(
    description="Run web-based ESV visualisation tool",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("esvs_pkl", type=Path, help="Path to extracted ESVs")
parser.add_argument("dataset_root", type=Path, help="Path dataset folder of videos")
parser.add_argument(
    "classes_csv", type=Path, help="Path to CSV containing name,id entries"
)
parser.add_argument(
    "--debug", action="store_true", help="Enable Dash debug capabilities"
)
parser.add_argument(
    "--port", default=8080, type=int, help="Port for webserver to listen on"
)
parser.add_argument("--host", default="localhost", help="Host to bind to")


def load_video(video_path: Union[str, Path]) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while capture.isOpened():
        success, frame = capture.read()
        if success:
            frames.append(frame[..., ::-1])  # BGR -> RGB
        else:
            break
    if len(frames) == 0:
        raise ValueError(f"Could not load video from {video_path}")

    return np.stack(frames)


@dataclass
class Result:
    esvs: List[np.ndarray]  # [n_frames_idx][frame_idx, class_idx]
    scores: np.ndarray  # [n_frames_idx, class_idx]
    uid: str
    label: int
    sequence_idxs: List[np.ndarray]  # [n_frames_idx][frame_idx]
    results_idx: int

    @property
    def max_n_frames(self):
        return max([len(s) for s in self.sequence_idxs])


class ShapleyValueResults:
    def __init__(self, results):
        self._results = results

    @property
    def uids(self) -> List[str]:
        return list(self._results["uids"])

    @property
    def shapley_values(self) -> List[np.ndarray]:
        # shapley_values[n_frames_idx][example_idx, frame_idx, class_idx]
        return self._results["shapley_values"]

    @property
    def sequence_idxs(self) -> np.ndarray:
        # sequence_idxs[n_frames_idx][example_idx]
        return self._results["sequence_idxs"]

    @property
    def labels(self) -> np.ndarray:
        return self._results["labels"]

    @property
    def scores(self) -> np.ndarray:
        # sequence_idxs[n_frames_idx, example_idx, class_idx]
        return self._results["scores"]

    @property
    def max_n_frames(self) -> int:
        return len(self._results["scores"])

    @cached_property
    def available_classes(self) -> List[int]:
        return sorted(np.unique(self.labels))

    @cached_property
    def class_counts(self) -> Dict[int, int]:
        return pd.Series(self.labels).value_counts().to_dict()

    @cached_property
    def class_example_idxs_lookup(self) -> Dict[int, np.ndarray]:
        return {
            cls: np.nonzero(self.labels == cls)[0] for cls in self.available_classes
        }

    def __getitem__(self, idx: Union[int, str]):
        if isinstance(idx, (int, np.int32, np.int64)):
            example_idx = idx
        elif isinstance(idx, str):
            example_idx = self.uids.index(idx)
        else:
            raise ValueError(f"Cannot handle idx type: {idx.__class__.__name__}")

        return Result(
            esvs=[esvs[example_idx] for esvs in self.shapley_values],
            scores=self.scores[:, example_idx],
            uid=self.uids[example_idx],
            label=self.labels[example_idx],
            sequence_idxs=[
                sequence_idxs[example_idx] for sequence_idxs in self.sequence_idxs
            ],
            results_idx=example_idx,
        )


def get_triggered_props():
    ctx = dash.callback_context
    return {trigger["prop_id"] for trigger in ctx.triggered}


class Visualisation:
    def __init__(
        self,
        results: ShapleyValueResults,
        class2str: Dict[int, str],
        dataset_dir: Path,
        title: str = "ESV Dashboard",
    ):
        self.results = results
        self.class2str = class2str
        self.str2class = {v: k for k, v in class2str.items()}
        self.dataset_dir = dataset_dir
        self.title = title

        def decode_other_classes(classes_str):
            return list(map(int, classes_str.split(":")))

        self.default_state = {
            "n-frames": self.results.max_n_frames,
            "uid": self.results.uids[0],
            "selected-classes": [],
        }

        self.state_types = {
            "uid": str,
            "n-frames": int,
            "selected-classes": decode_other_classes,
        }

    def extract_state_from_url(self, url):
        components = urlparse(url)
        query_string = parse_qsl(components.query)
        state = self.default_state.copy()
        for k, v in query_string:
            state[k] = self.state_types[k](v)
        return state

    def load_result(self, cls, example_idx):
        return self.results[self.results.class_example_idxs_lookup[cls][example_idx]]

    def attach_to_app(self, app: Dash):
        def app_layout():
            return html.Div(
                [dcc.Location(id="url", refresh=False), self.render_layout()]
            )

        app.layout = app_layout
        self.attach_callbacks(app)
        self.attach_routes(app)

    def attach_routes(self, app: Dash):
        @app.server.route("/videos/<uid>")
        def load_video(uid: str):
            path = self.dataset_dir / f"{uid}.webm"
            return flask.send_from_directory(self.dataset_dir.absolute(), f"{uid}.webm")

        @app.server.route("/frames/<uid>/<int:frame_idx>")
        def load_frame(uid: str, frame_idx: int):
            vid = self.load_video(uid)
            frame = vid[frame_idx]
            success, frame_jpeg = cv2.imencode(".jpg", frame[..., ::-1])
            response = make_response(frame_jpeg.tobytes())
            response.headers.set("Content-Type", "image/jpeg")
            response.headers.set(
                "Content-Disposition", "attachment", filename=f"{uid}-{frame_idx}.jpg"
            )
            return response

    def get_cls_and_example_idx_for_uid(self, uid):
        cls = self.results.labels[self.results.uids.index(uid)]
        uids = np.array(self.results.uids)
        class_uids = self.results.class_example_idxs_lookup[cls]
        example_idx = list(uids[class_uids]).index(uid)
        return cls, example_idx

    def get_uid_from_cls_and_example_idx(self, cls, example_idx):
        return np.array(self.results.uids)[self.results.class_example_idxs_lookup[cls]][
            example_idx
        ]

    def get_preds_df(self, result: Result, n_frames: int):
        scores = result.scores[n_frames - 1]
        classes = list(scores.argsort()[::-1][:10])
        if result.label not in classes:
            classes = classes[:-1] + [result.label]
        entries = []
        for i, cls in enumerate(classes):
            class_name = (
                self.class2str[cls]
                .replace("something", "[...]")
                .replace("Something", "[...]")
            )
            # We have to truncate labels on the x-axis so that they fit without all
            # getting horribly cut off
            max_len = 33
            truncated_class_name = class_name
            if len(class_name) >= max_len:
                truncated_class_name = class_name[: max_len - len("...")] + "..."

            entries.append(
                {
                    "Idx": i,
                    "Class": class_name,
                    "TruncatedClass": truncated_class_name,
                    "ClassId": cls,
                    "Score": scores[cls],
                }
            )

        return pd.DataFrame(entries)

    def attach_callbacks(self, app: Dash):
        @app.callback(
            Output("class-dropdown", "value"),
            Input("url", "href"),
        )
        def update_class_dropdown_value(href):
            state = self.parse_state_from_url(href)
            if "uid" not in state:
                raise PreventUpdate
            cls, _ = self.get_cls_and_example_idx_for_uid(state["uid"])
            return cls

        @app.callback(
            Output("n-frames-slider", "value"),
            Input("url", "href"),
        )
        def update_n_frames(href):
            state = self.parse_state_from_url(href)
            if "n-frames" not in state:
                raise PreventUpdate
            return state["n-frames"]

        @app.callback(
            Output("example-idx-slider", "value"),
            Input("class-dropdown", "value"),
            Input("url", "href"),
        )
        def update_example_slider_value(cls, href):
            ctx = dash.callback_context
            url_trigger = "url.href" in get_triggered_props()
            state = self.parse_state_from_url(href)
            if url_trigger and "uid" in state:
                _, example_idx = self.get_cls_and_example_idx_for_uid(state["uid"])
                return example_idx
            return 0

        @app.callback(
            Output("example-idx-slider", "max"),
            Output("example-idx-slider", "disabled"),
            Output("example-idx-slider", "marks"),
            Input("class-dropdown", "value"),
        )
        def update_example_slider(cls):
            max_index = self.results.class_counts[cls] - 1
            marks = {i: str(i) for i in range(max_index + 1)}
            return max_index, max_index == 0, marks

        @app.callback(
            Output("model-preds-bar", "clickData"),
            Output("model-preds-bar", "figure"),
            Input("class-dropdown", "value"),
            Input("example-idx-slider", "value"),
            Input("n-frames-slider", "value"),
        )
        def update_scores(cls, example_idx, n_frames):
            result = self.get_result(cls, example_idx)
            return None, self.plot_preds(self.get_preds_df(result, n_frames))

        @app.callback(
            Output("state-uid", "children"),
            Input("class-dropdown", "value"),
            Input("example-idx-slider", "value"),
        )
        def update_uid(cls, example_idx):
            idx = self.results.class_example_idxs_lookup[cls][example_idx]
            return self.results.uids[idx]

        @app.callback(
            Output("esv-scatter", "figure"),
            Input("state-uid", "children"),
            Input("n-frames-slider", "value"),
            Input("state-alt-class", "children"),
        )
        def update_esvs(uid, n_frames, alt_class_str):
            try:
                alt_class = int(alt_class_str)
            except ValueError:
                alt_class = None

            result = self.results[uid]
            return self.plot_esvs(result, n_frames, alt_class=alt_class)

        @app.callback(
            Output("esv-scatter", "hoverData"), Input("n-frames-slider", "value")
        )
        def update_esv_scatter_hover_data(_):
            return None

        @app.callback(
            Output("state-alt-class", "children"),
            Input("model-preds-bar", "clickData"),
            Input("state-uid", "children"),
        )
        def update_selected_classes(clickData, uid):
            if "state-uid" in get_triggered_props():
                return ""

            if clickData is not None:
                cls = clickData["points"][0]["customdata"][0]
                return str(cls)
            return dash.no_update

        @app.callback(
            Output("current-frame-container", "children"),
            Input("state-uid", "children"),
            Input("esv-scatter", "hoverData"),
        )
        def update_selected_frame(uid, hoverData):
            result = self.results[uid]
            if hoverData is None or "state-uid.children" in get_triggered_props():
                frame_index = 0
            else:
                frame_index = hoverData["points"][0]["x"]
            return html.Img(src=f"/frames/{result.uid}/{frame_index}")

        @app.callback(
            Output("video-container", "children"),
            Input("state-uid", "children"),
        )
        def update_video(uid):
            return html.Video(src=f"/videos/{uid}", loop=True, autoPlay=True)

        @app.callback(
            Output("url", "search"),
            [
                Input("example-idx-slider", "value"),
                Input("class-dropdown", "value"),
                Input("n-frames-slider", "value"),
            ],
        )
        def update_url_params(example_idx, cls, n_frames):
            state = {
                "uid": self.get_uid_from_cls_and_example_idx(cls, example_idx),
                "n-frames": n_frames,
            }
            params = urlencode(state)
            return f"?{params}"

    def render_layout(self):
        idx = self.results.uids.index(self.default_state["uid"])
        cls = self.results.labels[idx]
        available_example_idxs = list(self.results.class_example_idxs_lookup[cls])
        example_idx = available_example_idxs.index(idx)
        return html.Div(
            [
                html.Div(html.H1(self.title)),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Class: "),
                                dcc.Dropdown(
                                    id="class-dropdown",
                                    options=[
                                        {
                                            "label": self.class2str[cls],
                                            "value": cls,
                                        }
                                        for cls in self.results.available_classes
                                    ],
                                    value=cls,
                                ),
                            ],
                            className="control-element",
                        ),
                        html.Div(
                            [
                                html.Label("Example: "),
                                dcc.Slider(
                                    id="example-idx-slider",
                                    min=0,
                                    max=len(available_example_idxs),
                                    disabled=False,
                                    value=example_idx,
                                ),
                            ],
                            className="control-element",
                        ),
                        html.Div(
                            [
                                html.Label("Frames fed to model: "),
                                dcc.Slider(
                                    id="n-frames-slider",
                                    min=1,
                                    max=self.results.max_n_frames,
                                    marks={
                                        i: str(i)
                                        for i in range(1, self.results.max_n_frames + 1)
                                    },
                                    value=self.results.max_n_frames,
                                ),
                            ],
                            className="control-element",
                        ),
                    ],
                    className="controls",
                ),
                html.Hr(),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Model Predictions"),
                                dcc.Graph(
                                    id="model-preds-bar",
                                    config={"displayModeBar": False},
                                    responsive=True,
                                ),
                            ],
                            id="model-preds-bar-container",
                        ),
                        html.Div(
                            [
                                html.H2("ESV Values"),
                                dcc.Graph(
                                    id="esv-scatter",
                                    config={"displayModeBar": False},
                                    responsive=True,
                                    # if we don't set the initial height of the graph it
                                    # gets a height of 0 before it is updated when
                                    # the user clicks on an alternate class which
                                    # refreshes the height attribute of the Graph div.
                                    style={"height": "450px"},
                                ),
                            ],
                            id="esv-scatter-container",
                        ),
                    ],
                    id="graph-pane",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Hovered Frame:"),
                                html.Div(
                                    id="current-frame-container",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("Orignal Video:"),
                                html.Div(
                                    id="video-container",
                                ),
                            ]
                        ),
                    ],
                    id="video-pane",
                ),
                html.A(
                    target="_blank",
                    href="https://www.youtube.com/watch?v=zoUJi6L6z0M&feature=youtu.be",
                    children=html.Div(id="help-btn", children=html.Div("?")),
                ),
                html.Div(
                    id="state-uid",
                    children=self.default_state["uid"],
                    style={"display": "none"},
                ),
                html.Div(id="state-alt-class", children="", style={"display": "none"}),
            ],
            id="visualisation",
        )

    def plot_esvs(self, result: Result, n_frames: int, alt_class: Optional[int] = None):
        classes = [result.label]
        if alt_class is not None and alt_class != result.label:
            classes.append(alt_class)

        entries = []
        for cls in classes:
            for i in range(n_frames):
                entries.append(
                    {
                        "Segment": i + 1,
                        "Frame": result.sequence_idxs[n_frames - 1][i],
                        "ESV": result.esvs[n_frames - 1][i, cls],
                        "Class": self.class2str[cls]
                        + ("" if cls != result.label else " (GT)"),
                    }
                )
        df = pd.DataFrame(entries)
        figure = px.line(
            df,
            x="Frame",
            y="ESV",
            color="Class",
            line_shape="spline",
        )
        figure.update_traces(mode="markers+lines")
        figure.update_layout(
            margin_r=0,
            margin_b=20,
            hovermode="x unified",
            legend={"yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            transition={"duration": 400, "easing": "cubic-in-out"},
        )
        figure.add_hline(y=0)
        return figure

    def plot_preds(self, df):
        figure = px.bar(
            df,
            x="Idx",
            y="Score",
            hover_data={"Idx": False, "Class": True, "Score": True},
            custom_data=["ClassId"],
            labels={"Idx": ""},
        )
        figure.update_traces(marker_color="red")
        figure.update_layout(margin_b=200, margin_r=0)
        figure.update_xaxes(
            tickmode="array",
            tickvals=df.Idx,
            ticktext=df.TruncatedClass,
            tickangle=90,
            automargin=True,
        )
        return figure

    @lru_cache(maxsize=10)
    def load_video(self, uid: str) -> np.ndarray:
        return load_video(self.dataset_dir / f"{uid}.webm")

    def get_result(self, cls: int, example_idx: int) -> Result:
        idx = self.results.class_example_idxs_lookup[cls][example_idx]
        return self.results[idx]

    def parse_state_from_url(self, url):
        components = urlparse(url)
        query_string = parse_qsl(components.query)
        state = dict()
        for k, v in query_string:
            state[k] = self.state_types[k](v)
        return state


args = parser.parse_args()
dataset_dir: Path = args.dataset_root

classes = pd.read_csv(args.classes_csv, index_col="name")["id"]
class2str = {class_id: name for name, class_id in classes.items()}

results_dict = pd.read_pickle(args.esvs_pkl)
result_attributes = results_dict.get("attrs", {})

title = "ESV Dashboard"
if "dataset" in result_attributes:
    title += f" - {result_attributes['dataset']}"
if "model" in result_attributes:
    title += f" - {result_attributes['model']}"

results = ShapleyValueResults(results_dict)
visualisation = Visualisation(results, class2str, dataset_dir, title=title)

app = Dash(
    __name__,
    title="ESV Visualiser",
    update_title="Updating..." if args.debug else None,
    external_stylesheets=[dbc.themes.COSMO],
)
visualisation.attach_to_app(app)

if __name__ == "__main__":
    app.run_server(host=args.host, debug=args.debug, port=args.port)
else:
    print("Running in wsgi mode")
    application = app.server
