import json
import evaluator

if __name__ == "__main__":
    print(json.dumps({"fraction": evaluator.evaluate_hw()}))