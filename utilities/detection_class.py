"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import numpy as np
from utilities import base_class
from dtaidistance import dtw


class AnomalyDetector(base_class.BaseProcessor):
    def __init__(
        self,
        model_path: str = "",
        window_size: int | None = None,
        sampling_rate: int | None = None,
        original_sampling_rate: int | None = None,
        calculate_delay: bool | None = None,
        reverse_window_penalty: bool = True,
        label_keyword: str = "normal",
        mislabel_prob: float = 0.0,
        budget: int | None = None,
    ) -> None:
        """
        This class comprises all required functions to evaluate the anomaly detection performance of a given model.

        :param model_path: path to the trained model
        :param window_size: window size
        :param sampling_rate: sampling rate of input signal
        :param original_sampling_rate: sampling rate of the original data
        :param calculate_delay: boolean indicating to calculate delay
        :param reverse_window_penalty: boolean indicating to apply reverse window penalty
        :param label_keyword: label to identify nominal data
        :param mislabel_prob: probability of mislabeling
        :param budget: number of queries to be selected per split
        """

        super().__init__()
        self.model_path = model_path
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.original_sampling_rate = original_sampling_rate
        self.calculate_delay = calculate_delay
        self.reverse_window_penalty = reverse_window_penalty
        self.label_keyword = label_keyword
        self.mislabel_prob = mislabel_prob
        self.budget = budget
        self.dtw_lookup_table = None

    @staticmethod
    def unsupervised_threshold(
        detection_score_list: list[np.ndarray],
    ) -> float:
        """
        This function calculates the unsupervised threshold.

        :param detection_score_list: list of detection scores, each of shape (number_of_timesteps, channels)
        :return: threshold
        """

        assert isinstance(detection_score_list, list), (
            "detection_score_list argument must be a list!"
        )
        assert all(
            isinstance(detection_score, np.ndarray)
            for detection_score in detection_score_list
        ), "All items in detection_score_list must be numpy arrays!"

        max_detection_scores = []
        for detection_score in detection_score_list:
            if len(detection_score.shape) == 2:
                detection_score = detection_score.sum(axis=-1)
            max_detection_scores.append(np.max(detection_score))
        return np.max(np.array(max_detection_scores)).astype(float)

    def _find_detection_delay(
        self,
        predicted_anomaly_start: int,
        sequence_length: int,
        groundtruth_anomaly_start: int,
    ) -> float:
        """
        This function calculates the total detection delay for a given reverse window mode.

        :param predicted_anomaly_start: time step of predicted anomaly start
        :param sequence_length: sequence length
        :param groundtruth_anomaly_start: time step of groundtruth anomaly start
        """

        assert self.window_size is not None, "window_size must be provided!"
        assert self.sampling_rate is not None, "sampling_rate must be provided!"

        # If detection time step is before sequence_length - window_size
        if predicted_anomaly_start < sequence_length - self.window_size:
            rev_window_penalty = self.window_size
        # If detection time step is within last window_size time steps
        else:
            rev_window_penalty = sequence_length - predicted_anomaly_start
        # Sum detection delay with reverse window delay penalty and subtract in case of SS anomaly, then convert to seconds
        delay = abs(
            (predicted_anomaly_start + rev_window_penalty - groundtruth_anomaly_start)
            / self.sampling_rate
        )
        return delay

    def extract_groundtruth(
        self,
        input_list: list[np.ndarray],
    ) -> tuple[list[bool], list[int]]:
        """
        This function extracts the groundtruth labels and start times from the file names.

        :param input_list: list of multivariate time series, each of shape (number_of_timesteps, channels)
        """

        assert isinstance(input_list, list), "input_list must be a list!"
        assert all(isinstance(input_array, np.ndarray) for input_array in input_list), (
            "All items in input_list must be numpy arrays!"
        )
        assert self.sampling_rate is not None, "sampling_rate must be provided!"
        assert self.original_sampling_rate is not None, (
            "original_sampling_rate must be provided!"
        )

        groundtruth_labels = [
            self.label_keyword not in data_ts.dtype.metadata["file_name"]
            for _, data_ts in enumerate(input_list)
        ]
        groundtruth_start_list = [
            int(data_ts.dtype.metadata["file_name"].split("_")[-1].split(".")[0])
            // (self.original_sampling_rate // self.sampling_rate)
            for _, data_ts in enumerate(input_list)
        ]
        return groundtruth_labels, groundtruth_start_list

    def evaluate_online(
        self,
        detection_score_list: list[np.ndarray],
        threshold: float,
        input_list: list[np.ndarray] | None = None,
        groundtruth_list: tuple[list[bool], list[int]] | None = None,
    ) -> tuple[list[bool], list[int], list[float]]:
        """
        This function evaluates the anomaly detection performance of a given model.

        :param detection_score_list: list of detection scores, each of shape (number_of_timesteps, channels)
        :param threshold: detection threshold
        :param input_list: list of multivariate time series, each of shape (number_of_timesteps, channels)
        :param groundtruth_list: list of groundtruth labels, each a tuple containing the binary label and the first anomalous time step
        """

        assert isinstance(detection_score_list, list), (
            "detection_score_list must be a list!"
        )
        assert all(
            isinstance(detection_score, np.ndarray)
            for detection_score in detection_score_list
        ), "All items in detection_score_list must be numpy arrays!"
        assert isinstance(threshold, float), "threshold must be a float!"
        assert input_list is not None or groundtruth_list is not None, (
            "input_list or groundtruth_list must be provided!"
        )
        assert input_list is None or groundtruth_list is None, (
            "Only provide input_list or groundtruth_list!"
        )

        assert self.sampling_rate is not None, "sampling_rate must be provided!"
        assert self.original_sampling_rate is not None, (
            "original_sampling_rate must be provided!"
        )

        if input_list is not None and groundtruth_list is None:
            assert isinstance(input_list, list), "input_list must be a list!"
            assert all(
                isinstance(input_array, np.ndarray) for input_array in input_list
            ), "All items in input_list must be numpy arrays!"
            assert all(input_array.ndim == 2 for input_array in input_list), (
                "All items in input_list must be 2D numpy arrays!"
            )
            groundtruth_labels, groundtruth_start_list = self.extract_groundtruth(
                input_list
            )
        elif input_list is None and groundtruth_list is not None:
            assert isinstance(groundtruth_list, tuple), (
                "groundtruth_list must be a list!"
            )
            assert all(
                isinstance(groundtruth, list) for groundtruth in groundtruth_list
            ), (
                "All items in input_list must be a Tuple containing the binary label and the first anomalous time step!"
            )
            groundtruth_labels, groundtruth_start_list = groundtruth_list

        total_delays = []
        predicted_labels = []
        for idx_detection_score, detection_score in enumerate(detection_score_list):
            if len(detection_score.shape) == 2:
                detection_score = detection_score.sum(axis=-1)
            # Ground-truth normal time series
            if not groundtruth_labels[idx_detection_score]:
                # >0 time steps in anomaly score higher than threshold
                # False positive
                if np.sum(detection_score >= threshold) > 0:
                    predicted_labels.append(True)
                # =0 time steps in anomaly score higher than threshold
                # True negative
                else:
                    predicted_labels.append(False)
            # Ground-truth anomalous time series
            else:
                # Extract groundtruth anomaly start from file name and correct it for lower sampling rate
                groundtruth_start = int(groundtruth_start_list[idx_detection_score])
                # >0 time steps in anomaly score higher than threshold
                # Anomaly predicted
                if np.sum(detection_score >= threshold) > 0:
                    predicted_anomaly_start = np.argwhere(detection_score >= threshold)[
                        0
                    ][0]
                    # First predicted anomalous time step is after the groundtruth anomaly start
                    # True positive
                    if predicted_anomaly_start >= groundtruth_start:
                        predicted_labels.append(True)
                    # First predicted anomalous time step is before the groundtruth anomaly start
                    # False positive
                    else:
                        predicted_labels.append(
                            np.NaN
                        )  # Append np.Nan to indicate change to groundtruth labels
                    if self.calculate_delay:
                        if self.reverse_window_penalty:
                            delay = self._find_detection_delay(
                                predicted_anomaly_start,
                                len(detection_score),
                                groundtruth_start,
                            )
                        else:
                            delay = (
                                abs(predicted_anomaly_start - groundtruth_start)
                                / self.sampling_rate
                            )
                        total_delays.append(delay)
                # =0 time steps in anomaly score higher than threshold
                # False negative
                else:
                    predicted_labels.append(False)
                    if self.calculate_delay:
                        delay = (
                            len(detection_score) - groundtruth_start
                        ) / self.sampling_rate
                        total_delays.append(delay)

        groundtruth_labels_corrected, predicted_labels_corrected = self._correct_labels(
            groundtruth_labels, predicted_labels
        )

        return groundtruth_labels_corrected, predicted_labels_corrected, total_delays

    @staticmethod
    def _correct_labels(
        groundtruth_labels: list[bool],
        predicted_labels: list[int],
    ) -> tuple[list[bool], list[int]]:
        """
        This method finds false positives due to premature positive predictions and corrects the corresponding labels in groundtruth_labels and predicted_labels.

        :param groundtruth_labels: List of groundtruth labels
        :param predicted_labels: List of predicted labels
        """

        assert isinstance(groundtruth_labels, list), (
            "groundtruth_labels must be a list!"
        )
        assert all(
            isinstance(groundtruth_label, bool)
            for groundtruth_label in groundtruth_labels
        ), "All items in groundtruth_labels must be booleans!"
        assert isinstance(predicted_labels, list), "predicted_labels must be a list!"

        nan_indices = np.where(np.isnan(predicted_labels))[0]
        groundtruth_labels_corrected = groundtruth_labels.copy()
        predicted_labels_corrected = predicted_labels.copy()
        for nan_idx in nan_indices:
            assert groundtruth_labels_corrected[nan_idx] is True
            groundtruth_labels_corrected[nan_idx] = False
            predicted_labels_corrected[nan_idx] = True
        return groundtruth_labels_corrected, predicted_labels_corrected

    def corrupt_labels(
        self,
        groundtruth_list: tuple[list, list],
    ) -> tuple[list, list]:
        """
        Flip labels randomly with a given probability.

        :param groundtruth_list:
        """

        groundtruth_labels, groundtruth_start_list = groundtruth_list

        contaminated_labels = []
        contaminated_start_list = []
        for label, start in zip(groundtruth_labels, groundtruth_start_list):
            if np.random.binomial(1, self.mislabel_prob) == 0:
                contaminated_labels.append(label)
                contaminated_start_list.append(start)
            else:
                contaminated_labels.append(not label)
                contaminated_start_list.append(0.0)
        return contaminated_labels, contaminated_start_list

    def random_query_strategy(
        self,
        query_list: list[np.ndarray],
        query_score_list: list[np.ndarray],
        candidate_list: list[np.ndarray],
        candidate_score_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method implements the random-based query strategy.

        :param query_list: list of selected multivariate time series, each of shape (number_of_timesteps, channels)
        :param query_score_list: list of selected anomaly scores corresponding to query_list
        :param candidate_list: list of multivariate time series to be selected, each of shape (number_of_timesteps, channels)
        :param candidate_score_list: list of selected anomaly scores corresponding to candidate_list
        """

        assert isinstance(query_list, list), "query_list must be a list!"
        assert all(isinstance(query_ts, np.ndarray) for query_ts in query_list), (
            "All items in query_list must be numpy arrays!"
        )
        assert all(query_ts.ndim == 2 for query_ts in query_list), (
            "All items in query_list must be 2D numpy arrays!"
        )
        assert isinstance(query_score_list, list), "query_score_list must be a list!"
        assert all(
            isinstance(query_score, np.ndarray) for query_score in query_score_list
        ), "All items in query_score_list must be numpy arrays!"
        assert isinstance(candidate_list, list), "candidate_list must be a list!"
        assert all(
            isinstance(candidate_ts, np.ndarray) for candidate_ts in candidate_list
        ), "All items in candidate_list must be numpy arrays!"
        assert all(candidate_ts.ndim == 2 for candidate_ts in candidate_list), (
            "All items in candidate_list must be 2D numpy arrays!"
        )
        assert isinstance(candidate_score_list, list), (
            "candidate_score_list must be a list!"
        )
        assert all(
            isinstance(candidate_score, np.ndarray)
            for candidate_score in candidate_score_list
        ), "All items in candidate_score_list must be numpy arrays!"

        assert self.budget is not None, "budget must be provided!"

        for _ in range(self.budget):
            selection_idx = np.random.randint(0, high=len(candidate_score_list))
            query_list.append(candidate_list[selection_idx])
            query_score_list.append(candidate_score_list[selection_idx])
            del candidate_list[selection_idx]
            del candidate_score_list[selection_idx]
        return query_list, query_score_list, candidate_list, candidate_score_list

    def uncertainty_query_strategy(
        self,
        query_list: list[np.ndarray],
        query_score_list: list[np.ndarray],
        candidate_list: list[np.ndarray],
        candidate_score_list: list[np.ndarray],
        threshold: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method implements the uncertainty-based query strategy.

        :param query_list: list of selected multivariate time series, each of shape (number_of_timesteps, channels)
        :param query_score_list: list of selected anomaly scores corresponding to query_list
        :param candidate_list: list of multivariate time series to be selected, each of shape (number_of_timesteps, channels)
        :param candidate_score_list: list of selected anomaly scores corresponding to candidate_list
        :param threshold: initial detection threshold
        """

        assert isinstance(query_list, list), "query_list must be a list!"
        assert all(isinstance(query_ts, np.ndarray) for query_ts in query_list), (
            "All items in query_list must be numpy arrays!"
        )
        assert all(query_ts.ndim == 2 for query_ts in query_list), (
            "All items in query_list must be 2D numpy arrays!"
        )
        assert isinstance(query_score_list, list), "query_score_list must be a list!"
        assert all(
            isinstance(query_score, np.ndarray) for query_score in query_score_list
        ), "All items in query_score_list must be numpy arrays!"
        assert isinstance(candidate_list, list), "candidate_list must be a list!"
        assert all(
            isinstance(candidate_ts, np.ndarray) for candidate_ts in candidate_list
        ), "All items in candidate_list must be numpy arrays!"
        assert all(candidate_ts.ndim == 2 for candidate_ts in candidate_list), (
            "All items in candidate_list must be 2D numpy arrays!"
        )
        assert isinstance(candidate_score_list, list), (
            "candidate_score_list must be a list!"
        )
        assert all(
            isinstance(candidate_score, np.ndarray)
            for candidate_score in candidate_score_list
        ), "All items in candidate_score_list must be numpy arrays!"
        assert isinstance(threshold, float), "threshold must be a float!"

        assert self.budget is not None, "budget must be provided!"

        for _ in range(self.budget):
            selection_idx = np.argmin(
                abs(
                    np.array(
                        [
                            np.max(candidate_score)
                            for candidate_score in candidate_score_list
                        ]
                    )
                    - threshold
                )
            )
            query_list.append(candidate_list[selection_idx])
            query_score_list.append(candidate_score_list[selection_idx])
            del candidate_list[selection_idx]
            del candidate_score_list[selection_idx]
        return query_list, query_score_list, candidate_list, candidate_score_list

    def top_query_strategy(
        self,
        query_list: list[np.ndarray],
        query_score_list: list[np.ndarray],
        candidate_list: list[np.ndarray],
        candidate_score_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method implements the top-based query strategy.

        :param query_list: list of selected multivariate time series, each of shape (number_of_timesteps, channels)
        :param query_score_list: list of selected anomaly scores corresponding to query_list
        :param candidate_list: list of multivariate time series to be selected, each of shape (number_of_timesteps, channels)
        :param candidate_score_list: list of selected anomaly scores corresponding to candidate_list
        """

        assert isinstance(query_list, list), "query_list must be a list!"
        assert all(isinstance(query_ts, np.ndarray) for query_ts in query_list), (
            "All items in query_list must be numpy arrays!"
        )
        assert all(query_ts.ndim == 2 for query_ts in query_list), (
            "All items in query_list must be 2D numpy arrays!"
        )
        assert isinstance(query_score_list, list), "query_score_list must be a list!"
        assert all(
            isinstance(query_score, np.ndarray) for query_score in query_score_list
        ), "All items in query_score_list must be numpy arrays!"
        assert isinstance(candidate_list, list), "candidate_list must be a list!"
        assert all(
            isinstance(candidate_ts, np.ndarray) for candidate_ts in candidate_list
        ), "All items in candidate_list must be numpy arrays!"
        assert all(candidate_ts.ndim == 2 for candidate_ts in candidate_list), (
            "All items in candidate_list must be 2D numpy arrays!"
        )
        assert isinstance(candidate_score_list, list), (
            "candidate_score_list must be a list!"
        )
        assert all(
            isinstance(candidate_score, np.ndarray)
            for candidate_score in candidate_score_list
        ), "All items in candidate_score_list must be numpy arrays!"

        assert self.budget is not None, "budget must be provided!"

        for _ in range(self.budget):
            selection_idx = np.argmax(
                [np.max(candidate_score) for candidate_score in candidate_score_list]
            )
            query_list.append(candidate_list[selection_idx])
            query_score_list.append(candidate_score_list[selection_idx])
            del candidate_list[selection_idx]
            del candidate_score_list[selection_idx]
        return query_list, query_score_list, candidate_list, candidate_score_list

    def dissimilarity_query_strategy(
        self,
        query_list: list[np.ndarray],
        query_score_list: list[np.ndarray],
        candidate_list: list[np.ndarray],
        candidate_score_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method implements the dissimilarity-based query strategy.

        :param query_list: list of selected multivariate time series, each of shape (number_of_timesteps, channels)
        :param query_score_list: list of selected anomaly scores corresponding to query_list
        :param candidate_list: list of multivariate time series to be selected, each of shape (number_of_timesteps, channels)
        :param candidate_score_list: list of selected anomaly scores corresponding to candidate_list
        """

        assert isinstance(query_list, list), "query_list must be a list!"
        assert all(isinstance(query_ts, np.ndarray) for query_ts in query_list), (
            "All items in query_list must be numpy arrays!"
        )
        assert all(query_ts.ndim == 2 for query_ts in query_list), (
            "All items in query_list must be 2D numpy arrays!"
        )
        assert isinstance(query_score_list, list), "query_score_list must be a list!"
        assert all(
            isinstance(query_score, np.ndarray) for query_score in query_score_list
        ), "All items in query_score_list must be numpy arrays!"
        assert isinstance(candidate_list, list), "candidate_list must be a list!"
        assert all(
            isinstance(candidate_ts, np.ndarray) for candidate_ts in candidate_list
        ), "All items in candidate_list must be numpy arrays!"
        assert all(candidate_ts.ndim == 2 for candidate_ts in candidate_list), (
            "All items in candidate_list must be 2D numpy arrays!"
        )
        assert isinstance(candidate_score_list, list), (
            "candidate_score_list must be a list!"
        )
        assert all(
            isinstance(candidate_score, np.ndarray)
            for candidate_score in candidate_score_list
        ), "All items in candidate_score_list must be numpy arrays!"

        assert self.budget is not None, "budget must be provided!"

        for _ in range(self.budget):
            if len(query_score_list) == 0:
                # Random selection for the first budget iteration in '1day' split
                selection_idx = np.random.randint(0, high=len(candidate_score_list))
            else:
                # Compute distance matrix between query and candidates
                query_distances = np.array(
                    [
                        [
                            self.compute_dtw_distance(query_score, candidate_score)
                            for candidate_score in candidate_score_list
                        ]
                        for query_score in query_score_list
                    ]
                )
                # Find the most similar candidate to any query
                min_distance_idx = np.argmin(query_distances.mean(axis=0))
                sim_candidate = candidate_score_list[min_distance_idx]
                # Compute distance of all candidates to the most similar candidate
                sim_candidate_distances = np.array(
                    [
                        self.compute_dtw_distance(sim_candidate, candidate_score)
                        for candidate_score in candidate_score_list
                    ]
                )
                # Select the candidate most dissimilar to the most similar candidate
                selection_idx = np.argmax(sim_candidate_distances)
            # Update query lists and remove selected candidate
            query_list.append(candidate_list[selection_idx])
            query_score_list.append(candidate_score_list[selection_idx])
            del candidate_list[selection_idx]
            del candidate_score_list[selection_idx]
        return query_list, query_score_list, candidate_list, candidate_score_list

    def compute_dtw_distance(
        self,
        score_1: np.ndarray,
        score_2: np.ndarray,
    ) -> float:
        """
        This method computes the DTW distance between two time series if has not been computed before.

        :param score_1:
        :param score_2:
        """

        assert isinstance(score_1, np.ndarray), "score_1 must be a np.ndarray!"
        assert score_1.ndim == 1, "score_1 must be 1D numpy array!"
        assert isinstance(score_2, np.ndarray), "score_2 must be a np.ndarray!"
        assert score_2.ndim == 1, "score_2 must be 1D numpy array!"

        if self.dtw_lookup_table is None:
            self.dtw_lookup_table = {}

        key = tuple(
            sorted(
                (
                    score_1.dtype.metadata["file_name"],
                    score_2.dtype.metadata["file_name"],
                )
            )
        )
        # If distance for time series pair is not in look-up table, compute and store it
        if key not in self.dtw_lookup_table:
            self.dtw_lookup_table[key] = dtw.distance_fast(
                score_1.astype(np.dtype(np.double)), score_2.astype(np.dtype(np.double))
            )
        return self.dtw_lookup_table[key]
