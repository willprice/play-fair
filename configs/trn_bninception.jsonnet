{
    local frame_count = 8,
    local class_count = 174,
    frame_samplers: {
        local kind = "TemporalSegmentSampler",

        train: {
            kind: kind,
            frame_count: frame_count,
            test_mode: false
        },
        test: {
            kind: kind,
            frame_count: frame_count,
            test_mode: true
        }
    },
    transform: {
        preserve_aspect_ratio: true,
        image_scale_factor: 1 / 0.875,
        train: {
            hflip: false
        }
    },
    dataset: {
        kind: "SomethingSomethingV2Dataset",
        root: std.extVar("PROJECT_ROOT") + "/datasets/ssv2/gulp",
        class_count: class_count
    },
    model: {
        kind: "TRN",
        backbone_settings: {
            input_size: [3, 224, 224],
            input_order: 'CHW',
            input_space: 'BGR',
            input_range: [0, 255],
            mean: [104, 117, 128],
            std: [1, 1, 1],
        },
        class_count: class_count,
        backbone: "bninception",
        backbone_dim: 256,
        hidden_dim: 256,
        n_hidden_layers: 1,
        dropout: 0.7,
        frame_count: frame_count,
        batch_norm: false,
        backbone_checkpoint: std.extVar("PROJECT_ROOT") + "checkpoints/backbones/trn.pth",
        temporal_module_checkpoint: std.extVar("PROJECT_ROOT") + "checkpoints/features/trn_8_frames.pth",
    }
}