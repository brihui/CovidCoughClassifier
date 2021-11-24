import os
import ffmpeg
import json


def convert(dataset_path, filename):
    # Change to the dataset directory
    # print(dataset_path)
    # current_directory = os.getcwd().__str__()
    # print(current_directory)
    # data_directory = current_directory
    # print(data_directory)
    # os.chdir(data_directory)
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
    # Change back to the current directory
    # os.chdir(os.getcwd())


def get_labeled_coughs(path):
    """
    Takes the path to a folder of CoughVid data and returns a dictionary of
     the sample's health status and the .webm file names for the labeled data
    """
    labeled_coughs = []
    num_jsons = 0
    covid = 0
    symptomatic = 0
    healthy = 0
    webm = 0
    ogg = 0

    # Loop through all files
    for file in os.listdir(path):
        # Only open the JSONs
        if file.endswith(".json"):
            num_jsons += 1
            with open(f"{path}/{file}", 'r') as f:
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
                        with open(f"{path}/{file}", 'r') as webm_f:
                            # Add the cough audio file to good coughs
                            labeled_coughs.append({"status": status, "file_name": webm_file})
                            webm += 1
                    except IOError:
                        labeled_coughs.append({"status": status, "file_name": ogg_file})
                        ogg += 1
    print('Total number of samples: ', num_jsons)
    print('Healthy: ', healthy)
    print('Symptomatic: ', symptomatic)
    print('COVID: ', covid)
    print('Webm: ', webm)
    print('Ogg: ', ogg)
    # print(labeled_coughs)
    return labeled_coughs


def main():
    dataset_path = "./coughvid_dataset"
    change_path = "/coughvid_dataset"

    labeled_cough_dict = get_labeled_coughs(dataset_path)

    for cough_dict in labeled_cough_dict:
        convert(change_path, cough_dict["file_name"])


if __name__ == "__main__":
    main()
