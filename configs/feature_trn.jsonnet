{
    local frame_count = 8,
    local class_count = 174,
    frame_samplers: {
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
    dataset: {
        kind: "SomethingSomethingV2FeaturesDataset",
        path: std.extVar("PROJECT_ROOT") + "/datasets/ssv2/features/trn.hdf",
        class_count: class_count,
        in_memory: true,
    },
    model: {
        kind: "TRN",
        frame_count: frame_count,
        class_count: class_count,
        input_dim: 256,
        hidden_dim: 256,
        n_hidden_layers: 1,
        dropout: 0.1,
        batch_norm: false,
        input_relu: false,
        checkpoint: std.format(std.extVar("PROJECT_ROOT") + "/checkpoints/features/trn_%s_frames.pth", frame_count),
    }
}