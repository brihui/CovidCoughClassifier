import json
import os
import matplotlib.pyplot as plt
import shutil
# for loading and visualizing audio files
import librosa
import librosa.display


def coughvid_to_spectro():
    """
    Converts coughvid WAVs to spectrograms
    """
    # Step 1: Check data
    coughvid_path = os.getcwd() + "/coughvid_dataset/"
    # Get all the coughvid wavs in a list and json labels in a list
    # indexes of two lists match
    cough_wavs, json_files = get_coughvid_wavs()
    print("No. of .wav files in coughvid folder = ", len(cough_wavs))
    index_start = 0

    # Step 2: Load audio file and visualize its waveform (using librosa)
    # default sr is 22050
    for index in range(index_start, len(cough_wavs)):
        x, sr = librosa.load(coughvid_path+cough_wavs[index], sr=22050)
        # print(type(x), type(sr))  # <class 'numpy.ndarray'> <class 'int'>

        # plt.figure(figsize=(14, 5))
        librosa.display.waveplot(x, sr=sr)
        # plt.show()

        # Convert the audio waveform to spectrogram
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        # plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        # plt.colorbar()
        # plt.show()


def get_coughvid_wavs():
    """
    Accesses the coughvid folder and returns a List of the wav files
    :return: List of wav files and list of json files
    """
    # Move into coughvid dataset
    os.chdir(os.getcwd() + "/coughvid_dataset")

    wav_list = []
    json_list = []

    for file in os.listdir():
        if file.endswith('.wav'):
            wav_list.append(file)
        elif file.endswith('.json'):
            json_list.append(file)

    return wav_list, json_list


def get_coswara_spectrograms():
    """
    Accesses the coswara folder and saves all wavs as spectrograms
    :return: List of wav files and list of json files
    """
    spectrogram_directory = os.getcwd() + "/spectrograms/coswara"

    # Move into coughvid dataset
    coswara_folder = os.getcwd() + "/coswara_dataset"
    os.chdir(coswara_folder)

    heavy_cough_list = []
    shallow_cough_list = []
    metadata_list = []
    corrupted_list = []

    save_file_name = 1

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
                        # Add cough into corresponding list
                        if file.endswith("heavy.wav"):
                            heavy_cough_list.append(file)
                            # Create spectrogram in this function
                            try:
                                x, sr = librosa.load(os.getcwd() + "/" + file, sr=22050)

                                # print(type(x), type(sr))  # <class 'numpy.ndarray'> <class 'int'>

                                # plt.figure(figsize=(14, 5))
                                librosa.display.waveplot(x, sr=sr)
                                plt.show()

                                # Convert the audio waveform to spectrogram
                                X = librosa.stft(x)
                                Xdb = librosa.amplitude_to_db(abs(X))
                                # plt.figure(figsize=(14, 5))
                                librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                                # plt.colorbar()
                                # plt.show()

                                # Determine covid status
                                with open(os.getcwd() + "/metadata.json", 'r') as f:
                                    data = json.load(f)
                                    covid_status = data["covid_status"]

                                # If covid is positive, store as 1, else store 0
                                if covid_status.startswith('positive'):
                                    # plt.savefig(spectrogram_directory + "/1_" + save_file_name.__str__() + ".jpg")
                                    save_file_name += 1
                                else:
                                    # plt.savefig(spectrogram_directory + "/0_" + save_file_name.__str__() + ".jpg")
                                    save_file_name += 1

                            except ValueError:
                                print(participant + "wav file corrupted")
                                corrupted_list.append(participant)
                        elif file.endswith("shallow.wav"):
                            shallow_cough_list.append(file)
                        elif file.startswith("metadata"):
                            metadata_list.append(file)
                    os.chdir(folder_directory)
        os.chdir(coswara_folder)

    print('Num heavy coughs: ', len(heavy_cough_list))
    print('Num shallow coughs: ', len(shallow_cough_list))
    print('Num metadata: ', len(metadata_list))

    return heavy_cough_list, shallow_cough_list, metadata_list, corrupted_list


def get_coughvid_spectrograms():
    """
    Accesses the coughvid folder and saves all wavs as spectrograms
    :return: List of wav files and list of json files
    """
    spectrogram_directory = os.getcwd() + "/spectrograms/coughvid"

    # Move into coughvid dataset
    coughvid_folder = os.getcwd() + "/coughvid_dataset"
    os.chdir(coughvid_folder)

    cough_list = []
    json_list = []
    corrupted_list = []

    file_name = 1
    i = 0

    for file in os.listdir():
        i += 1
        print(i)
        # If file is the json file, add to the json list
        if file.endswith(".json"):
            json_list.append(file)
        elif file.endswith(".py"):
            pass
        else:
            cough_list.append(file)
            try:
                x, sr = librosa.load(os.getcwd() + "/" + file, sr=22050)

                # print(type(x), type(sr))  # <class 'numpy.ndarray'> <class 'int'>

                # plt.figure(figsize=(14, 5))
                librosa.display.waveplot(x, sr=sr)
                # plt.show()

                # Convert the audio waveform to spectrogram
                X = librosa.stft(x)
                Xdb = librosa.amplitude_to_db(abs(X))
                # plt.figure(figsize=(14, 5))
                librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                # plt.colorbar()

                # Determine covid status
                with open(file.split('.wav')[0] + ".json", "r") as f:
                    data = json.load(f)
                    covid_status = data["status"]

                if covid_status == "COVID-19":
                    plt.savefig(spectrogram_directory + "/1_" + file_name.__str__() + ".jpg")
                    plt.close()
                    file_name += 1
                else:
                    plt.savefig(spectrogram_directory + "/0_" + file_name.__str__() + ".jpg")
                    plt.close()
                    file_name += 1
                # plt.show()
            except ValueError:
                print(file + " wav file corrupted")
                corrupted_list.append(file)

    print('Num coughs: ', len(cough_list))
    print('Num corrupted coughs: ', len(corrupted_list))
    print('Num jsons: ', len(json_list))

    return cough_list, corrupted_list, json_list


def main():
    # get_coswara_spectrograms()
    get_coughvid_spectrograms()
    plt.close("all")


if __name__ == "__main__":
    main()
