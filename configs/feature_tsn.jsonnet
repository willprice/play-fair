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
        kind: "SomethingSomethingV2FeaturesDataset",
        path: std.extVar("PROJECT_ROOT") + "/datasets/ssv2/features/tsn.hdf",
        class_count: class_count,
        in_memory: true,
    },
    model: {
        kind: "TSN",
        class_count: class_count,
        input_dim: 256,
        dropout: 0.7,
        input_relu: true,
        checkpoint: std.extVar("PROJECT_ROOT") + "checkpoints/features/tsn.pth",
    }
}