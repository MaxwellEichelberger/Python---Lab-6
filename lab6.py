import argparse
import json
import numpy as np
from scipy.io.wavfile import write

# Constants
SAMPLE_RATE = 44100
FADE_DURATION = 0.01
MAX_PCM_VALUE = 32767
KEYS = ["A0", "A#0", "B0"] + \
       [note + str(octave) for octave in range(1, 8) for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]] + \
       ["C8"]


def scale_waveform(waveform):
    """Scale waveform to be within the 16-bit PCM range.

    This function takes a waveform and scales it to be within the range of a 16-bit PCM signal,
    which is from -32768 to 32767.

    Args:
        waveform (np.ndarray): The input waveform to be normalized.

    Returns:
        np.ndarray: The normalized waveform as a 16-bit integer array.
    """
    # TODO-7: BEGIN
    # Normalize the waveform provided to be between -1 and 1
    
    normalized_waveform = (waveform - np.min(waveform))/(np.max(waveform)-np.min(waveform))

    # TODO-7: END

    # Scale the normalized waveform to the maximum PCM value
    scaled_waveform = normalized_waveform * MAX_PCM_VALUE

    # TODO-8: BEGIN
    # Return the scaled_waveform as a 16-bit integer array
    
    return scaled_waveform.astype(np.int16)

    # TODO-8: END


def create_stereo_waveform(lh_waveform, rh_waveform, sample_rate=44100):
    """Creates a normalized stereo waveform from left-hand and right-hand waveforms.

    Args:
        lh_waveform (np.ndarray): The waveform for the left hand.
        rh_waveform (np.ndarray): The waveform for the right hand.
        sample_rate (int, optional): The sample rate of the waveforms. Defaults to 44100.

    Returns:
        np.ndarray: The normalized stereo waveform.

    """
    # Ensure both waveforms are the same length
    # TODO-6: BEGIN
    # Pad the shorter waveform with zeros to match the length of the longer waveform

    if (lh_waveform.size < rh_waveform.size):
        num = rh_waveform.size - lh_waveform.size
        lh_waveform = np.pad(lh_waveform, (0,num), mode='constant', constant_values = 0)

    elif (rh_waveform.size < lh_waveform.size):
        num = lh_waveform.size - rh_waveform.size
        rh_waveform = np.pad(rh_waveform, (0,num), mode='constant', constant_values = 0)
    # TODO-6: END

    # Scale waveforms
    lh_waveform = scale_waveform(lh_waveform)
    rh_waveform = scale_waveform(rh_waveform)

    # Create stereo waveform, this is a 2-column array with the left hand waveform
    # in the first column and the right hand waveform in the second column.
    # TODO-9: BEGIN
    # Combine the left-hand and right-hand waveforms into a stereo waveform
    
    stereo_waveform = np.hstack((lh_waveform.reshape(-1,1),rh_waveform.reshape(-1,1)))
    # TODO-9: END

    return stereo_waveform


def generate_musical_note_waveform(note, duration, frequency_dict, sample_rate=SAMPLE_RATE):
    """Generates a waveform for a musical note with fade-in and fade-out.

    This function generates a waveform for a given musical note, applying a fade-in at the start
    and a fade-out at the end to prevent clicks at the note's boundaries.

    Args:
        note (str): The musical note to generate. Must be a key in `frequency_dict`.
        duration (float): The duration of the note in seconds.
        frequency_dict (dict): A dictionary mapping musical notes to their frequencies in Hertz.
        sample_rate (int, optional): The sample rate of the generated waveform. Defaults to SAMPLE_RATE.

    Returns:
        np.ndarray: The generated waveform as a NumPy array.

    Raises:
        ValueError: If `note` is not in `frequency_dict`.
    """
    if note not in frequency_dict:
        raise ValueError(
            f"The note {note} is not in the frequency dictionary.")

    # Generate a waveform for the given note
    # TODO-5: BEGIN
    # Create a NumPy array t representing time, ranging from 0 to duration with steps of 1 / sample_rate.
    
    t = np.array(np.arange(0, duration, 1/sample_rate).tolist())

    # Retrieve the frequency of the given musical note from the frequency_dict dictionary. Store this value in a variable named frequency.
    
    frequency = frequency_dict[note]

    # Generate the initial wveform as a sine wave. The formula for the waveform is 0.5 * np.sin(2 * np.pi * frequency * t). Store the generated waveform in a variable named waveform.
    
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)

    # TODO-5: END

    # Apply fade-in and fade-out
    fade_in_duration = int(sample_rate * FADE_DURATION)
    fade_out_duration = int(sample_rate * FADE_DURATION)

    fade_in = np.linspace(0, 1, fade_in_duration, endpoint=False)
    fade_out = np.linspace(1, 0, fade_out_duration, endpoint=False)

    waveform[:fade_in_duration] *= fade_in
    waveform[-fade_out_duration:] *= fade_out

    return waveform


def generate_hand_signal(notes, durations, frequency_dict, sample_rate=SAMPLE_RATE):
    """Generates a waveform for a sequence of musical notes.

    This function takes a sequence of musical notes and their durations, and generates
    a corresponding waveform by concatenating the waveforms of individual notes.

    Args:
        notes (list of str): A list of musical notes.
        durations (list of float): A list of durations for each note in seconds.
        frequency_dict (dict): A dictionary mapping musical notes to their frequencies in Hertz.
        sample_rate (int, optional): The sample rate of the generated waveform. Defaults to SAMPLE_RATE.

    Returns:
        np.ndarray: The generated waveform for the sequence of notes.

    Raises:
        ValueError: If the lengths of `notes` and `durations` lists are not equal.
    """
    if len(notes) != len(durations):
        raise ValueError(
            "The lengths of `notes` and `durations` lists must be equal.")

    signal = np.array([])
    for note, duration in zip(notes, durations):
        note_signal = generate_musical_note_waveform(
            note, duration, frequency_dict, sample_rate)
        signal = np.concatenate((signal, note_signal))

    return signal


def calculate_frequency(key_number):
    """Calculates the frequency of a musical note based on its key number.

    This function calculates the frequency of a musical note using the formula:
    frequency = 440 * 2 ** ((key_number - 49) / 12)

    Args:
        key_number (int): The key number of the musical note. For a standard 88-key piano,
                          the key numbers range from 1 for A0, to 88 for C8.

    Returns:
        float: The frequency of the musical note in Hertz (Hz).
    """
    return 440 * 2 ** ((key_number - 49) / 12)


def load_song_composition(file_path):
    """Loads a song composition from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the song composition.

    Returns:
        list: A list containing two lists, one for the left hand notes and durations,
               and one for the right hand notes and durations. Each list contains list
               of [note, duration].

    Raises:
        json.JSONDecodeError: If there is an error decoding the JSON file.
        KeyError: If the expected keys are not found in the JSON data.

    Example:
        >>> load_song_composition('sample_composition.json')
        ([['C4', 1], ['E4', 1], ['G4', 1]], [['E4', 1], ['G4', 1], ['C5', 1]])
    """
    try:
        # TODO-1: BEGIN
        # Load the JSON file specified by file_path into a Python dictionary

        file = open(file_path)
        JDict = json.load(file)

        # Return the left (left_hand) and right (right_hand) compositions from the dictionary
        return JDict['right_hand'], JDict['left_hand']
        # TODO-1: END

    except json.JSONDecodeError:
        print("Error decoding the JSON file.")
        raise
    except KeyError as e:
        print(f"Key error: {e}")
        raise
    # TODO-2: BEGIN
    # Handle exception if JSON file is not found
    except:
        print("JSON File is Not Found")
    # TODO-2: END


def main():
    """Generates a WAV file from a song composition.

    The function parses command line arguments for the input JSON file containing
    the song composition and the output path for the WAV file. It then generates
    the corresponding waveforms for the left and right hand compositions and
    saves them to the specified WAV file.
    """
    parser = argparse.ArgumentParser(
        description='Generate a WAV file from a song composition.'
    )
    parser.add_argument(
        'composition_file', type=str,
        help='Path to the JSON file with the song composition'
    )
    parser.add_argument(
        'wav_file', type=str,
        help='Path where the WAV file should be written to'
    )
    
    args = parser.parse_args()
    

    lh_composition, rh_composition = load_song_composition(args.composition_file)

    frequencies = {key: calculate_frequency(
        i + 1) for i, key in enumerate(KEYS)}

    # TODO-3: BEGIN
    # Unzip the left hand composition (lh_composition) and assign parts to the notes and durations variables.
    notes = []
    durations = []

    for i in range(len(lh_composition)):
        x = lh_composition[i][0]
        notes.append(x)
        y = lh_composition[i][1]
        durations.append(y)

    # TODO-3: END

    lh_waveform = generate_hand_signal(notes, durations, frequencies)

    # TODO-4: BEGIN
    # Unzip the right hand composition (rh_composition) and assign parts to the notes and durations variables.
    
    notes.clear()
    durations.clear()

    for i in range(len(rh_composition)):
        x = rh_composition[i][0]
        notes.append(x)
        y = rh_composition[i][1]
        durations.append(y)

    # TODO-4: END
    
    rh_waveform = generate_hand_signal(notes, durations, frequencies)

    stereo_waveform = create_stereo_waveform(lh_waveform, rh_waveform)

    write(args.wav_file, 44100, stereo_waveform)


if __name__ == "__main__":
    main()
