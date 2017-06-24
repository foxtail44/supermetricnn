# Configuration profiles
smcnn_conf = {
    "SM1":
        {
            "path_train":       "./dataset/train",
            "path_eval":        "./dataset/validate/",
            "learning_rate":    0.0001,
            "epochs":           100000,
            "batch_size":       10,
            "checkpoint_file":  "checkpoint.pth.tar",
            "model_best":       "model_best.pth.tar",
            "print_freq":       100
        }
}