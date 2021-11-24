import os
import ffmpeg
import json


def convert(filename):
    if filename.endswith('.webm'):
        command = "ffmpeg -i {} {}'.wav'".format(filename, filename[:-5])
        os.system(command)
        # delete original file, be careful
        command = 'rm {}'.format(filename)
        os.system(command)
    elif filename.endswith('.ogg'):
        command = "ffmpeg -i {} {}'.wav'".format(filename, filename[:-4])
        os.system(command)
        # delete original file, be careful
        command = 'rm {}'.format(filename)
        os.system(command)
    else:
        print("Error: unacceptable file extension", filename)


def get_labeled_coughs():
    """
    Looks through all files and finds the ones with 'status', returns the .webm
    or .ogg file name in a list as well as a list of dictionaries matching the
    status with each file name.
    """
    labeled_coughs = []
    labeled_coughs_dict = []
    num_jsons = 0
    covid = 0
    symptomatic = 0
    healthy = 0
    webm = 0
    ogg = 0

    # Loop through all files
    for file in os.listdir():
        # Only open the JSONs
        if file.endswith(".json"):
            num_jsons += 1
            with open(f"{file}", 'r') as f:
                file_data = json.load(f)
                # Check if the cough is labeled with 'status' - covid label
                if 'status' in file_data:
                    status = file_data['status']
                    # Count the different categories
                    if status == 'healthy':
                        healthy += 1
                    elif status == 'symptomatic':
                        symptomatic += 1
                    elif status == 'COVID-19':
                        covid += 1
                    # Try to get .webm audio file
                    audio_file_name = file.removesuffix(".json")
                    webm_file = audio_file_name + ".webm"
                    ogg_file = audio_file_name + ".ogg"
                    try:
                        with open(f"{webm_file}", 'r') as webm_f:
                            # Add the cough audio file to good coughs
                            labeled_coughs.append(webm_file)
                            labeled_coughs_dict.append({"status": status, "file_name": webm_file})
                            webm += 1
                    except IOError:
                        labeled_coughs.append(ogg_file)
                        labeled_coughs_dict.append({"status": status, "file_name": ogg_file})
                        ogg += 1
    print('Total number of samples: ', num_jsons)
    print('Healthy: ', healthy)
    print('Symptomatic: ', symptomatic)
    print('COVID: ', covid)
    print('Webm: ', webm)
    print('Ogg: ', ogg)
    print('Total labeled cough files: ', len(labeled_coughs))
    return labeled_coughs, labeled_coughs_dict


def count_wav_files():
    """
    Counts the number of WAV files and returns all wav files and their corresponding
    JSON files, as well as this python file to prevent it from being deleted.
    :return: 3 Lists
    """
    wavs = 0
    jsons = 0
    pys = 0

    wav_files = []
    json_files = []
    py_files = []

    for file in os.listdir():
        if file.endswith(".wav"):
            wav_files.append(file)
            wavs += 1
        elif file.endswith(".json"):
            json_files.append(file)
            jsons += 1
        elif file.endswith(".py"):
            py_files.append(file)
            pys += 1

    print('Wav files: ', wavs)
    print('JSON files: ', jsons)
    print('Python files: ', pys)
    return wav_files, json_files, py_files


def purge_unlabeled_data(wav_files, json_files, py_files):
    """
    Deletes all files in the directory not passed in
    :param wav_files: wav files labeled
    :param json_files: json files labeled
    :param py_files: this python file
    """
    num_purged = 0

    for file in os.listdir():
        if file not in wav_files and file not in json_files and file not in py_files:
            command = 'rm {}'.format(file)
            os.system(command)
            num_purged += 1

    print('Num files deleted: ', num_purged)


def main():
    labeled_cough, labeled_cough_dict = get_labeled_coughs()

    for cough_dict in labeled_cough_dict:
        convert(cough_dict["file_name"])

    wav_files, json_files, py_file = count_wav_files()

    purge_unlabeled_data(wav_files, json_files, py_file)


if __name__ == "__main__":
    main()
