import os
# import ffmpeg


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


audio_fpath = "./"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ", len(audio_clips))
# print(audio_clips[0])

for file in audio_clips:
    convert(file)
