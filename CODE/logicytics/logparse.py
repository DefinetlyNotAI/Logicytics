import re


def parse_weights_log(file_path, top_n=1) -> dict[str, list[dict[str, str]]]:
    with open(file_path, "r") as file:
        lines = file.readlines()

    results = []
    for line in lines:
        weights_match = re.search(r"Weights: ({.*?}), Average Accuracy: (\d+\.\d+)%", line)
        if weights_match:
            weights = eval(weights_match.group(1))
            accuracy = float(weights_match.group(2))
            results.append((accuracy, lines.index(line) + 1, weights))

    # Sort results by accuracy in descending order and get the top N results
    results.sort(reverse=True, key=lambda x: x[0])
    top_results = results[:top_n]

    return {
        "top_results": [
            {
                "accuracy": str(result_pwl[0]) if len(str(result_pwl[0])) == 5 else str(result_pwl[0]) + "0",
                "line": result_pwl[1],
                "weights": result_pwl[2]
            }
            for result_pwl in top_results
        ],
        "total_weights_tested": len(results)
    }


path = "weights_log.txt"
parsed_data = parse_weights_log(path, top_n=7)
for i, result in enumerate(parsed_data["top_results"], start=1):
    print(
        f"Top {i}:\n    Accuracy: {result['accuracy']}%,\n    Line: {result['line']},\n    Weights: {result['weights']}\n")
print("-" * 40)
print(f"\nTotal Weights Tested: {parsed_data['total_weights_tested']}\n")
print("-" * 40)

# Output:
"""
Top 1:
    Accuracy: 51.78%,
    Line: 1,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.0, "enhanced": 0.0, "levenshtein": 1.0}

Top 2:
    Accuracy: 51.60%,
    Line: 12,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.1, "enhanced": 0.0, "levenshtein": 0.9}

Top 3:
    Accuracy: 51.50%,
    Line: 22,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.2, "enhanced": 0.0, "levenshtein": 0.8}

Top 4:
    Accuracy: 51.40%,
    Line: 30,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.3, "enhanced": 0.0, "levenshtein": 0.7}

Top 5:
    Accuracy: 51.30%,
    Line: 37,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.4, "enhanced": 0.0, "levenshtein": 0.6}

Top 6:
    Accuracy: 51.19%,
    Line: 44,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.5, "enhanced": 0.0, "levenshtein": 0.5}

Top 7:
    Accuracy: 51.09%,
    Line: 50,
    Weights: {"jaccard": 0.0, "cosine": 0.0, "synonym": 0.0, "difflib": 0.6, "enhanced": 0.0, "levenshtein": 0.4}

----------------------------------------

Total Weights Tested: 2486

----------------------------------------
"""
