import os


def remove_useless():
    """
    Removes all files which aren't coughs
    """
    coswara_folder = os.getcwd()

    num_to_remove = 0

    # Loop through all date groups
    for folder in os.listdir():
        # If folder is any of these, don't do anything
        if folder == ".DS_Store" or folder == "coswara_preprocess.py":
            pass
        else:
            # Change directory into the folder
            os.chdir(os.getcwd() + "/" + folder)
            # Loop through each participant in date group
            for participant in os.listdir():
                if participant == ".DS_Store" or participant.endswith('.csv'):
                    pass
                else:
                    folder_directory = os.getcwd()
                    os.chdir(os.getcwd() + "/" + participant)
                    # Loop through each file in the participant
                    for file in os.listdir():
                        # If file doesn't start with cough or metadata, it's not useful to us
                        if not file.startswith("cough") and not file.startswith("metadata") and not file.endswith(".py"):
                            num_to_remove += 1
                            command = 'rm {}'.format(file)
                            os.system(command)
                    os.chdir(folder_directory)
        os.chdir(coswara_folder)

    print('Num files removed: ', num_to_remove)


def main():
    remove_useless()


if __name__ == "__main__":
    main()
