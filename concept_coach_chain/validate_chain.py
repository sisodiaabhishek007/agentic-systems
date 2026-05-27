from build_chain import build_chain


def is_response_valid(response: str) -> tuple[bool, list[str]]:
    errors = []

    # Check type
    if not isinstance(response, str):
        errors.append("Response is not a string.")

    # Check empty
    if isinstance(response, str) and not response.strip():
        errors.append("Response is empty.")

    # Check word limit
    if isinstance(response, str):
        word_count = len(response.split())
        if word_count > 100:
            errors.append(f"Response exceeds 100 words ({word_count} words).")

    return (len(errors) == 0, errors)


def run_tests():
    chain = build_chain()

    test_cases = [
        {
            "topic": "LangChain Expression Language",
            "analogy_domain": "school assembly line"
        },
        {
            "topic": "Prompt Templates",
            "analogy_domain": "wedding invitation cards"
        },
        {
            "topic": "Output Parsers",
            "analogy_domain": "food delivery packaging"
        }
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {idx} ---")
        print("Input:", test)

        try:
            response = chain.invoke(test)
        except Exception as e:
            response = f"Error during generation: {str(e)}"

        print("\nGenerated Response:")
        print(response)

        is_valid, errors = is_response_valid(response)

        print("\nValidation Result:", is_valid)

        if not is_valid:
            print("Errors:")
            for err in errors:
                print("-", err)


if __name__ == "__main__":
    run_tests()
