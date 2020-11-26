{
    local class_count = 174,
    frame_samplers: {
        local frame_count = 8,
        local kind = "RandomSampler",

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
        class_count: class_count,
    },
    model: {
        kind: "TSN",
        backbone_settings: {
            input_size: [3, 224, 224],
            input_order: 'CHW',
            input_space: 'RGB',
            input_range: [0, 1],
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        },
        backbone: "resnet50",
        class_count: class_count,
        backbone_dim: 256,
        dropout: 0.7,
        input_relu: true,
        backbone_checkpoint: std.extVar("PROJECT_ROOT") + "checkpoints/backbones/tsn.pth",
        temporal_module_checkpoint: std.extVar("PROJECT_ROOT") + "checkpoints/features/tsn.pth",
    }
}