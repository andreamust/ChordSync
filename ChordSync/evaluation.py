"""
Evaluation with mir_eval of the synchronisation performance of the system.
"""

import numpy as np
import torch
import torch.nn as nn
from mir_eval import alignment, chord
from scipy.signal import find_peaks
from torchaudio.functional import forced_align, merge_tokens
from torchaudio.models.decoder import ctc_decoder
from utils.chord_utils import ChordDecoder, Encoding


class EvaluateChord:
    """
    Evaluates the audio chord esitmation (ACE) using the chord module of mir_eval.
    The class is used to evaluate the performances of a Deep Learning model and
    to work with pytorch tensors.
    """

    def __init__(
        self,
        chord_prediction: torch.Tensor,
        chord_onsets: torch.Tensor,
        root_prediction: torch.Tensor | None = None,
        mode_prediction: torch.Tensor | None = None,
        majmin_prediction: torch.Tensor | None = None,
        chord_target: torch.Tensor | None = None,
        root_target: torch.Tensor | None = None,
        mode_target: torch.Tensor | None = None,
        majmin_target: torch.Tensor | None = None,
        blank: int = 0,
    ) -> None:
        """
        Initialize the class.
        """
        # set the blank token
        self.blank = blank

        # tensor of shape (batch_size, n_frames, n_chords)
        self.chord_prediction = chord_prediction.permute(0, 2, 1)
        self.root_prediction = (
            root_prediction.permute(0, 2, 1) if root_prediction else None
        )
        self.mode_prediction = (
            mode_prediction.permute(0, 2, 1) if mode_prediction else None
        )
        self.majmin_prediction = (
            majmin_prediction.permute(0, 2, 1) if majmin_prediction else None
        )

        # tensor of shape (batch_size, n_frames)
        self.chord_target = chord_target
        self.chord_onsets = chord_onsets.detach().to("cpu")


class EvaluateAlignment:
    """
    Evaluates the audio-to-score chord alignment using the alignment module of
    mir_eval.
    The class is used to evaluate the performances of a Deep Learning model and
    to work with pytorch tensors.
    """

    def __init__(
        self,
        blank: int = 0,
        window_size: float = 0.5,
        audio_length: int = 15,
        threshold: float = 0.7,
    ) -> None:
        """
        Initialize the class.
        """
        # set the blank token
        self.blank = blank
        # set the window size
        self.window_size = window_size
        # set the audio length
        self.audio_length = audio_length
        # set the threshold
        self.threshold = threshold
        # set the batch size
        self.batch_size = 0

    def evaluate_boundaries(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """ """
        # get batch size
        self.batch_size = predictions.shape[0]
        # tensor of shape (batch_size, n)
        targets = targets.type(torch.float32).detach().to("cpu").numpy()
        predictions = predictions.type(torch.float32).detach().to("cpu")
        # get sequence length
        sequence_length = predictions.shape[-1]
        # iterate over batches
        results_absolute_error, results_percentage_correct = 1, 0

        conuter = 0

        for i in range(self.batch_size):
            # get the actual values from chord prediction tensor
            chord_prediction = predictions[i]
            # get the actual values from chord onset tensor
            target = np.where(targets[i] == 1)[0]
            chord_onsets = self._unsample_timestamps(target, sequence_length)  # type: ignore
            chord_onsets = chord_onsets[chord_onsets != 0]

            peaks_number = len(chord_onsets)

            # find the peaks
            # peaks, peaks_attributes = find_peaks(
            #     chord_prediction, height=0, prominence=0, distance=1
            # )
            # best_prominences = np.argsort(peaks_attributes["prominences"])[
            #     :peaks_number
            # ]
            # # best peaks the ones with the best prominences
            # best_peaks = peaks[best_prominences]
            _, best_peaks = torch.topk(chord_prediction, peaks_number)
            # sort the peaks
            best_peaks = np.sort(best_peaks.numpy())

            # get the predicted timestamps
            predicted_timestamps = self._unsample_timestamps(
                best_peaks, sequence_length
            )

            # if i == 0:
            #     print(f"Chord onsets: {chord_onsets}")
            #     print(f"Predicted timestamps: {predicted_timestamps}")

            # no onsets in the ground truth and no peaks in the prediction
            if len(chord_onsets) == 0 and len(best_peaks) == 0:
                results_percentage_correct += 1
                results_absolute_error -= 1

            # no onsets in the ground truth
            elif len(chord_onsets) == 0:
                continue

            # normal case
            elif len(predicted_timestamps) == len(chord_onsets):
                # evaluate the system
                results_percentage_correct += self._mireval_percentage_correct(
                    predicted_timestamps=predicted_timestamps,
                    chord_onsets=chord_onsets,  # type: ignore
                )
                results_absolute_error -= self._mireval_absolute_error(
                    predicted_timestamps=predicted_timestamps,
                    chord_onsets=chord_onsets,  # type: ignore
                )
                conuter += 1

        # get the average results
        results_absolute_error /= self.batch_size
        results_percentage_correct /= self.batch_size

        # print(f"Counter: {conuter}")

        return results_absolute_error, results_percentage_correct

    def evaluate_with_boundaries(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        onsets: torch.Tensor,
    ) -> tuple:
        """ """
        # get batch size
        self.batch_size = predictions.shape[0]
        # tensor of shape (batch_size, n)
        onsets = onsets.detach().to("cpu").numpy()
        # get sequence length
        sequence_length = predictions.shape[-1]  # TODO: check this
        # iterate over batches
        results_absolute_error, results_percentage_correct = 0, 0

        for i in range(self.batch_size):
            # get the actual values from chord prediction tensor
            chord_prediction = predictions[i]
            chord_target = targets[i]
            # round the target to a certain threshold
            chord_target = torch.where(chord_target > self.threshold, 1, 0)
            chord_prediction = self.adjust_predictions(chord_prediction, chord_target)

            # get the actual values from chord onset tensor
            chord_onsets = onsets[i]
            chord_onsets = self._clean_onsets(chord_onsets)  # type: ignore

            # if the number of ones in chord_target != len(chord_onsets[i]) -> chord_target[0] = 1
            chord_prediction = self._check_prediction(
                chord_prediction, torch.tensor(chord_onsets)
            )

            # get indexes of ones in prediction
            prediction_index = self._get_boundary_index(chord_prediction)
            prediction_index = prediction_index.detach().to("cpu").numpy()

            # unsample prediction indexes
            predicted_timestamps = self._unsample_timestamps(
                prediction_index, sequence_length
            )

            try:
                # evaluate the system
                results_percentage_correct += self._mireval_percentage_correct(
                    predicted_timestamps=predicted_timestamps,
                    chord_onsets=chord_onsets,  # type: ignore
                )
                results_absolute_error += self._mireval_absolute_error(
                    predicted_timestamps=predicted_timestamps,
                    chord_onsets=chord_onsets,  # type: ignore
                )
            except:
                continue

        # get the average results
        results_absolute_error /= self.batch_size
        results_percentage_correct /= self.batch_size

        return results_absolute_error, results_percentage_correct

    def evaluate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        onsets: torch.Tensor,
    ) -> tuple:
        """
        Evaluate the system.

        Returns
        -------
        evaluation_results: tuple
            Tuple containing the evaluation results as floats.
        """
        # get batch size
        self.batch_size = predictions.shape[0]
        # tensor of shape (batch_size, n_frames, n_chords)
        predictions = predictions.permute(0, 2, 1)
        # tensor of shape (batch_size, n)
        onsets = onsets.detach().to("cpu").numpy()
        # get sequence length
        sequence_length = predictions.shape[-1]  # TODO: check this
        # get the lengths of the prediction and target tensors
        prediction_lengths, target_length = self._calculate_lengths(
            predictions, targets
        )
        # initialize metrics
        results_absolute_error, results_percentage_correct = 0, 0

        # iterate over batches
        for i in range(self.batch_size):
            # get the actual values from chord prediction tensor
            chord_prediction = predictions[i]
            chord_target = targets[i]

            # if chord_target is all zeros, skip the evaluation
            if torch.all(chord_target == 0):
                continue

            # batch the prediction and target tensors
            chord_prediction = self._batch_tensor(chord_prediction)
            chord_target = self._batch_tensor(chord_target)

            # get the actual values from chord onset tensor
            chord_onsets = onsets[i]
            chord_onsets = self._clean_onsets(chord_onsets)

            # forced alignment
            length_out = (prediction_lengths[i].unsqueeze(0),)
            length_target = (target_length[i].unsqueeze(0),)
            # get predicted timestamps from forced alignment
            predicted_timestamps = self._forced_alignment(
                chord_prediction, chord_target, length_out[0], length_target[0]
            )
            # unsample the timestamps
            predicted_timestamps = self._unsample_timestamps(
                predicted_timestamps, sequence_length
            )

            # evaluate the system
            results_percentage_correct += self._mireval_percentage_correct(
                predicted_timestamps=predicted_timestamps, chord_onsets=chord_onsets
            )
            results_absolute_error += self._mireval_absolute_error(
                predicted_timestamps=predicted_timestamps, chord_onsets=chord_onsets
            )

        # get the average results
        results_absolute_error /= self.batch_size
        results_percentage_correct /= self.batch_size

        return results_absolute_error, results_percentage_correct

    def _clean_onsets(self, onsets: torch.Tensor) -> np.ndarray:
        """
        Clean the onset tensor by removing the zeros and converting to numpy.

        Parameters
        ----------
        onsets: torch.Tensor
            Tensor containing the onset times.

        Returns
        -------
        onsets: np.ndarray
            Array containing the onset times.
        """
        # remove zero padding (preserve zeros if at index 0)
        return np.array([o for i, o in enumerate(onsets) if o != 0 or i == 0])

    def _batch_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Batch the tensor.

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor containing the values.

        Returns
        -------
        tensor: torch.Tensor
            Tensor containing the values.
        """
        # batch the tensor
        return tensor.unsqueeze(0)

    def _mireval_absolute_error(
        self,
        predicted_timestamps: np.ndarray,
        chord_onsets: np.ndarray,
        mode: str = "mae",
    ) -> float:
        """
        Evaluate the system with onset metrics.

        Parameters
        ----------
        predicted_timestamps: np.ndarray
            Array containing predicted timestamps.
        chord_onsets: np.ndarray
            Array containing the chord onsets.
        mode: str
            A modality between:
                - Median absolute error
                - Average absolute error

        Returns
        -------
        evaluation_results: float
            Evaluation results as floats.
        """
        # check mode
        assert mode in ["mae", "aae"]

        # get the percentage of correct chord onsets
        mae, aae = alignment.absolute_error(chord_onsets, predicted_timestamps)

        return mae if mode == "mae" else aae

    def _mireval_percentage_correct(
        self, predicted_timestamps: np.ndarray, chord_onsets: np.ndarray
    ) -> float:
        """
        Calculate the percentage of correct chord onsets.

        Parameters
        ----------
        predicted_timestamps: np.ndarray
            Array containing the predicted timestamps.
        chord_onsets: np.ndarray
            Array containing the chord onsets.

        Returns
        -------
        percentage_correct: float
            Percentage of correct chord onsets.
        """
        # get the percentage of correct chord onsets
        percentage_correct = alignment.percentage_correct(
            chord_onsets, predicted_timestamps, window=self.window_size
        )
        return percentage_correct

    def _unsample_timestamps(
        self, timestamps: np.ndarray, sequence_length: int
    ) -> np.ndarray:
        """
        Unsample the timestamps.

        Parameters
        ----------
        timestamps: np.ndarray
            Array containing the timestamps.
        sequence_length: int
            Length of the sequence.

        Returns
        -------
        timestamps: np.ndarray
            Array containing the timestamps.
        """
        assert sequence_length > 0, "Sequence length must be greater than 0."
        # unsample the timestamps
        unsampled = [(i * self.audio_length) / 323 for i in timestamps]
        return np.array(unsampled)

    def _forced_alignment(
        self,
        chord_prediction: torch.Tensor,
        chord_ground_truth: torch.Tensor,
        length_prediction: torch.Tensor,
        length_target: torch.Tensor,
    ) -> np.ndarray:
        """
        Forced alignment of the predicted chord sequence to the ground truth
        chord sequence.

        Parameters
        ----------
        chord_prediction: torch.Tensor
            Tensor containing the predicted chord sequence.
        chord_ground_truth: torch.Tensor
            Tensor containing the ground truth chord sequence.

        Returns
        -------
        predicted_timestamps: np.ndarray
            Array containing the predicted timestamps.
        """
        # predictions to log probabilities
        chord_prediction = nn.functional.log_softmax(chord_prediction, dim=-1)
        # remove zeros from ground truth
        chord_ground_truth = chord_ground_truth[chord_ground_truth != 0].unsqueeze(0)

        # get the forced alignment
        labels, scores = forced_align(
            chord_prediction,
            chord_ground_truth.int(),
            input_lengths=length_prediction,
            target_lengths=length_target,
            blank=self.blank,
        )
        # merge tokens
        labels = merge_tokens(labels.squeeze(), scores.squeeze(), blank=self.blank)

        # get the predicted timestamps
        predicted_timestamps = [i.start for i in labels]

        return np.array(predicted_timestamps)

    def _check_ones_count(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> bool:
        """
        Utility function that returns true if a tensor contains the number of
        ones of the target, false otherwise.
        """
        # print(f"{torch.count_nonzero(predictions)} - {torch.count_nonzero(targets)}")
        return bool(torch.count_nonzero(predictions) == torch.count_nonzero(targets))

    def _get_boundary_index(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Utility function that returns the indexes of the ones in a tensor.

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor containing the values.

        Returns
        -------
        indexes: torch.Tensor
            Tensor containing the indexes.
        """
        return torch.nonzero(tensor, as_tuple=True)[0]

    def _check_prediction(
        self, prediction: torch.Tensor, onsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Utility function that checks if the number of ones in the prediction
        tensor is equal to the number of onsets in the ground truth tensor.
        If not, the first element of the prediction tensor is set to 1.

        Parameters
        ----------
        prediction: torch.Tensor
            Tensor containing the values.
        onsets: torch.Tensor
            Tensor containing the onsets.

        Returns
        -------
        prediction: torch.Tensor
            Tensor containing the values.
        """
        # if the number of ones in chord_target != len(chord_onsets[i]) -> chord_target[0] = 1
        if torch.count_nonzero(prediction) != len(onsets):
            prediction[0] = 1
        return prediction

    def _calculate_lengths(
        self, chord_prediction: torch.Tensor, chord_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the lengths of the prediction and target tensors.

        Parameters
        ----------
        chord_prediction: torch.Tensor
            Tensor containing the predicted chord sequence.
        chord_target: torch.Tensor
            Tensor containing the ground truth chord sequence.

        Returns
        -------
        prediction_lengths: torch.Tensor
            Tensor containing the lengths of the prediction tensor.
        target_length: torch.Tensor
            Tensor containing the lengths of the target tensor.
        """
        prediction_lengths = torch.full(
            size=(self.batch_size,),
            fill_value=chord_prediction.shape[1],
            dtype=torch.long,
        )
        target_length = torch.count_nonzero(chord_target, dim=1)

        return prediction_lengths, target_length

    def adjust_predictions(self, predictions, target):
        # Thresholding
        binary_predictions = (predictions >= self.threshold).float()

        # Post-Processing
        predicted_count = torch.sum(binary_predictions).item()
        target_count = torch.sum(target).item()

        if predicted_count < target_count:
            # Identify top N predictions and set them to 1
            top_values, top_indices = torch.topk(
                predictions, k=int(target_count - predicted_count)
            )
            binary_predictions[top_indices] = 1
        elif predicted_count > target_count:
            # Identify bottom N predictions and set them to 0
            bottom_values, bottom_indices = torch.topk(
                predictions, k=int(predicted_count - target_count), largest=False
            )
            binary_predictions[bottom_indices] = 0

        return binary_predictions


if __name__ == "__main__":

    # Example ground truth and estimated chord annotations (in seconds)
    ground_truth_chords = [
        "N",
        "C:maj",
        "G:7",
        "C:maj",
        "G:maj",
        "C:maj",
        "G:maj",
        "C:maj",
    ]
    estimated_chords = ["N", "C:min", "N", "N", "G:maj", "C:maj", "G:maj", "C:maj"]
    ground_truth_intervals = np.array(
        [
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.5],
            [1.5, 2.0],
            [2.0, 2.5],
            [2.5, 3.0],
            [3.0, 3.5],
            [3.5, 4.0],
        ]
    )
    estimated_intervals = np.array(
        [
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.5],
            [1.5, 2.0],
            [2.0, 2.5],
            [2.5, 3.0],
            [3.0, 3.5],
            [3.5, 4.0],
        ]
    )

    evaluation = chord.evaluate(
        ground_truth_intervals,
        ground_truth_chords,
        estimated_intervals,
        estimated_chords,
    )
    print(evaluation)

    decoder = ChordDecoder()
    print(decoder.decode(44))

    test_target = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0]])
    test_prediction = torch.tensor([[0.3, 0.1, 0.12, 0.1, 0.11, 0.13, 0.3, 0.1]])

    # abc = EvaluateAlignment().evaluate_with_boundaries(
    #     test_prediction, test_target, torch.tensor([[0.0, 0.5, 1.0]])
    # )
    # print(abc)

    from scipy.signal import find_peaks

    # create a test tensor of values from 0 to 1 with a step of 0.001
    test = torch.tensor(
        [[0.3285, 0.9931, 0.9931, 0.2131, 0.9931, 0.2131, 0.2131, 0.2131, 0.2131]]
    )
