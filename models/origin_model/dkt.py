import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2017")
    parser.add_argument("--model_name", type=str, default="cdkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--sample_type", type=str, default="gumbel")
    parser.add_argument("--testing_sample", action="store_true", default=False)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    if args.model_name not in ["cdkt"]:
        del args.sample_type
        del args.testing_sample

    params = vars(args)
    main(params)