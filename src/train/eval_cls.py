import argparse, json
from sklearn.metrics import accuracy_score, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    preds = json.load(open(args.preds))
    labels = json.load(open(args.labels))

    print("Accuracy:", accuracy_score(labels, preds))
    print("F1:", f1_score(labels, preds))

if __name__ == "__main__":
    main()
