import os 

def rename_subfolders(directory):
    subfolders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    for i, folder in enumerate(subfolders, start=1):
        old_path = os.path.join(directory, folder)
        new_path = os.path.join(directory, str(i))

        try:
            os.rename(old_path, new_path)
            print(f"Renamed '{folder}' -> '{i}'")
        except Exception as e:
            print(f"Error renaming '{folder}': {e}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ").strip()

    # Normalize the path and automatically fix any potential path formatting issues
    directory = os.path.normpath(directory)    

    if os.path.exists(directory) and os.path.isdir(directory):
        rename_subfolders(directory)
    else:
        print("Invalid directory path.")

