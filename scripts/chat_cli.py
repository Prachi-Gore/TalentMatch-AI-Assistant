from hireflow.chat import handle_user_message


def main() -> None:
    session_id = None
    print("TalentMatch CLI. Press Ctrl+C to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        output = handle_user_message(user_input, session_id=session_id)
        session_id = output["session_id"]
        print(f"\nTalentMatch:\n{output['response']}")


if __name__ == "__main__":
    main()

