import warnings
import numpy as np
import scipy.sparse as sp

from sklearn.exceptions import NotFittedError
from sklearn.base import _fit_context
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.utils.fixes import _IS_32BIT
from sklearn.utils._param_validation import StrOptions, Interval, RealNotInt
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, _document_frequency


class BM25Transformer(
    OneToOneFeatureMixin,
    TransformerMixin,
    BaseEstimator,
    auto_wrap_output_keys=None,
):
    _parameter_constraints: dict = {
        "k1": [Interval(RealNotInt, 0, None, closed="left")],
        "b": [Interval(RealNotInt, 0, 1, closed="both")],
        "norm": [StrOptions({"l1", "l2"}), None],
        "use_idf": ["boolean"],
        "smooth_idf": ["boolean"],
        "sqrt_tf": ["boolean"],
        "sublinear_tf": ["boolean"],
    }
    
    def __init__(
        self, 
        *, 
        k1=1.2, # [1.2, 2.0]
        b=0.75,
        norm="l2", 
        use_idf=True, 
        smooth_idf=True,
        sqrt_tf=False,
        sublinear_tf=False,
    ):
        self.k1 = k1
        self.b = b
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sqrt_tf = sqrt_tf
        self.sublinear_tf = sublinear_tf
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in (np.float64, np.float32) else np.float64
        
        # average document length used for bm25
        avgdl = X.sum(axis=1).mean()
        self.avgdl = avgdl

        if self.use_idf:
            n_samples, _ = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, copy=False)

            # perform idf smoothing if required
            df += float(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            # `np.log` preserves the dtype of `df` and thus `dtype`.
            self.idf_ = np.log(n_samples / df) + 1.0
            
        return self
    
    def transform(self, X, copy=True):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            copy=copy,
            reset=False,
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=X.dtype)

        if self.sqrt_tf:
            np.sqrt(X.data, X.data)
        
        # bm25 weight
        dl = np.asarray(X.sum(axis=1)).ravel()
        dl = np.repeat(dl, np.diff(X.indptr))
        X.data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1.0

        if hasattr(self, "idf_"):
            # the columns of X (CSR matrix) can be accessed with `X.indices `and
            # multiplied with the corresponding `idf` value
            X.data *= self.idf_[X.indices]

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def _more_tags(self):
        return {
            "X_types": ["2darray", "sparse"],
            "preserves_dtype": [np.float64, np.float32],
        }


class BM25Vectorizer(CountVectorizer):
    
    _parameter_constraints: dict = {**CountVectorizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "k1": [Interval(RealNotInt, 0, None, closed="left")],
            "b": [Interval(RealNotInt, 0, 1, closed="both")],
            "norm": [StrOptions({"l1", "l2"}), None],
            "use_idf": ["boolean"],
            "smooth_idf": ["boolean"],
            "sqrt_tf": ["boolean"],
            "sublinear_tf": ["boolean"],
        }
    )
    
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float32,
        k1=1.2,
        b=0.75,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sqrt_tf=False,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        self.k1 = k1
        self.b = b
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sqrt_tf = sqrt_tf
        self.sublinear_tf = sublinear_tf
        
    @property
    def idf_(self):
        if not hasattr(self, "_bm25"):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute."
            )
        return self._bm25.idf_
    
    @idf_.setter
    def idf_(self, value):
        if not self.use_idf:
            raise ValueError("`idf_` cannot be set when `user_idf=False`.")
        if not hasattr(self, "_bm25"):
            self._bm25 = BM25Transformer(
                k1=self.k1,
                b=self.b,
                norm=self.norm,
                use_idf=self.use_idf,
                smooth_idf=self.smooth_idf,
                sqrt_tf=self.sqrt_tf,
                sublinear_tf=self.sublinear_tf,
            )
        self._validate_vocabulary()
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal to vocabulary size = %d"
                    % (len(value), len(self.vocabulary))
                )
        self._bm25.idf_ = value
        
    @property
    def avgdl(self):
        if not hasattr(self, "_bm25"):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute."
            )
        return self._bm25.avgdl
    
    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, raw_documents, y=None):
        self._check_params()
        self._warn_for_unused_params()
        self._bm25 = BM25Transformer(
            k1=self.k1,
            b=self.b,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sqrt_tf=self.sqrt_tf,
            sublinear_tf=self.sublinear_tf,
        )
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self
    
    def fit_transform(self, raw_documents, y=None):
        self._check_params()
        self._bm25 = BM25Transformer(
            k1=self.k1,
            b=self.b,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sqrt_tf=self.sqrt_tf,
            sublinear_tf=self.sublinear_tf,
        )
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self._bm25.transform(X, copy=False)
    
    def transform(self, raw_documents):
        check_is_fitted(self, msg="The BM25Vectorizer instance is not fitted")
        
        X = super().transform(raw_documents)
        return self._bm25.transform(X, copy=False)
    
    def _more_tags(self):
        return {"X_types": ["string"], "_skip_test": True}
