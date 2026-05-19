from app import AdaptiveDSLM

if __name__ == "__main__":
    print("\n================================")
    print("ADAPTIVE DATA SCIENCE MODEL")
    print("================================")

    file_path = input("\nEnter CSV dataset path: ")
    system = AdaptiveDSLM(file_path)
    system.run_pipeline()
