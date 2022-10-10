import numpy as np
from typing import Optional, Union
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


class SKNN:
    def __init__(self,
                 n_predictions: Optional[int] = 20,
                 k: Optional[int] = 500,
                 similarity: Optional[str] = 'cosine'):
        """
        parameters
        ----------
        n_predictions: int | default=20
            Total amount of recommendations to return per session

        k: int | default=500
            Total neighbors for the current session

        similarity: str | default='cosine'
            Similarity metric to use (cosine, jaccard, cityblock)
        """

        if similarity not in {'cosine', 'jaccard', 'cityblock'}:
            raise ValueError("Invalid metric")

        self._k = k
        self._n_predictions = n_predictions
        self._similarity = similarity

        self._sess_matrix = None
        self._item_index = None
        self._item_cont_idx_arr = None
        self._sort_samples = None
        self._ignore_products = {0}

    def fit(self,
            sessions: np.ndarray,
            sort_samples: Optional[Union[np.ndarray, list]] = None,
            disable_progress: bool = False) -> None:
        """
        parameters
        ----------
        sessions: np.ndarray
            Products for each session in the format:
            n_sessions x total_products_padded

        sort_samples: np.ndarray / list
            array of vectors that'll be used to sort the most similar sessions
            If empty, samples randomly

        disable_progress: bool | default=False
        """

        if sort_samples is not None:
            self._sort_samples = sort_samples
            # Argsort descending
            sort_criteria = np.argsort(sort_samples)[::-1]

            # Order sessions by date, closer dates come first
            sessions = sessions[sort_criteria]

        # Set total possible contiguous items.
        max_prod = np.max(sessions)
        self._item_cont_idx_arr = np.arange(0, max_prod + 1)

        # Allocate empty_session matrix
        sess_matrix = np.zeros((sessions.shape[0], max_prod + 1),
                               dtype=np.bool_)

        self._sess_matrix = sess_matrix
        self._item_index = dict()

        # Populate the matrix with items
        for i in tqdm(range(len(sessions)), disable=disable_progress):

            session_products = sessions[i][sessions[i] > 0]

            # populate item_index hash_table
            for prod in session_products:

                if prod in self._item_index.keys():
                    self._item_index[prod].add(i)
                else:
                    self._item_index[prod] = {i}

            # Populate Matrix (assume unique makes the search faster)
            # WARNING: using assume_unique on not unique lists returns pure garbage
            bool_sess = np.isin(self._item_cont_idx_arr,
                                session_products,
                                assume_unique=True)
            self._sess_matrix[i, :] = bool_sess

        # Remove unseen products
        ignore_products = self._item_cont_idx_arr[np.isin(self._item_cont_idx_arr,
                                                          self._item_index.keys(),
                                                          assume_unique=True,
                                                          invert=True)]
        self._ignore_products.update(ignore_products)
        self._sess_matrix = csr_matrix(self._sess_matrix)

    def predict(self, session: np.ndarray) -> np.ndarray:

        # Remove padding from input_session if any.
        input_session = session[session > 0]
        input_session_bool = np.isin(self._item_cont_idx_arr,
                                     input_session,
                                     assume_unique=True)

        # get candidate sessions
        candidates = set()
        for itm in input_session:
            current_item_sessions = self._item_index.get(itm)
            # Ignore unseen items
            if current_item_sessions is None:
                continue
            candidates.update(current_item_sessions)
        candidates = np.fromiter(candidates, dtype=np.int64)

        if self._sort_samples is not None:
            np.ndarray.sort(candidates)  # Sorting in-place

            # Candidates are filtered by recency
            filtered_candidates = candidates[:self._k]
        else:
            if self._k >= len(candidates) or self._k < 0:
                filtered_candidates = candidates
            else:
                random = np.random.choice(range(len(candidates)),
                                          size=self._k,
                                          replace=False)
                # Candidates filtered Randomly
                filtered_candidates = candidates[random][:self._k]

        # If there are no candidates return an array of zeros
        if len(filtered_candidates) == 0:
            return np.zeros(self._n_predictions)

        # Candidate sessions are created
        candidate_sessions_bool = self._sess_matrix[filtered_candidates]

        # Apply similarity function
        if self._similarity == 'jaccard':
            # Jaccard does not work w/sparse matrices (scipy)
            candidate_sessions_bool = candidate_sessions_bool.toarray()

        # sklearn supports calculating distances w/sparse matrices
        neighbors_similarity = pairwise_distances(input_session_bool[None, :],
                                                  candidate_sessions_bool,
                                                  metric=self._similarity)[0]

        # Calculate KNN Score
        # Sum neighbor in Neightbors of Sim(Sess, Neighbor) * (bool if item is present or not in sess)
        candidate_items_scores = candidate_sessions_bool.multiply(neighbors_similarity[:, None])
        # np.matrix -> np.array
        candidate_items_ranking = np.asarray(candidate_items_scores.sum(axis=0))[0]

        # Sort recommendations by score, remove 0, invalid & prods already in the session
        remove = {*self._ignore_products, *input_session}
        recommendations = np.argsort(-candidate_items_ranking)
        recommendations = recommendations[np.isin(recommendations, remove,
                                                  invert=True,
                                                  assume_unique=True)]
        topk_rec = recommendations[:self._n_predictions]

        return topk_rec

    def predict_batch(self, sessions):

        batch = np.zeros((sessions.shape[0], self._n_predictions), dtype=np.int64)
        # Provisory tqdm for progress visibility
        for n, session in tqdm(enumerate(sessions)):

            rec = self.predict(session)
            # Pad if we don't have enough recommendations
            if len(rec) < self._n_predictions:
                zero_len = self._n_predictions - len(rec)
                zero_pad = np.zeros(zero_len)
                rec = np.hstack((rec, zero_pad))

            batch[n, :] = rec

        return batch

    @property
    def session_matrix(self):
        return self._sess_matrix
