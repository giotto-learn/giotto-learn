"""Construct and handle Mapper pipelines."""
# License: GNU AGPLv3

from sklearn.pipeline import Pipeline

from .cluster import ParallelClustering
from .nerve import Nerve
from .utils._list_feature_union import ListFeatureUnion
from .utils.pipeline import transformer_from_callable_on_rows, identity

global_pipeline_params = ("memory", "verbose")
nodes_params = ("scaler", "filter_func", "cover")
clust_prepr_params = ("clustering_preprocessing",)
clust_params = ("clusterer", "n_jobs",
                "parallel_backend_prefer")
nerve_params = ("min_intersection", "store_edge_elements", "contract_nodes")
clust_prepr_params_prefix = "pullback_cover__"
nodes_params_prefix = "pullback_cover__map_and_cover__"
clust_params_prefix = "clustering__"
nerve_params_prefix = "nerve__"


class MapperPipeline(Pipeline):
    """Subclass of :class:`sklearn.pipeline.Pipeline` to deal with
    pipelines generated by :func:`~gtda.mapper.pipeline.make_mapper_pipeline`.

    The :meth:`set_params` method is modified from the corresponding method in
    :class:`sklearn.pipeline.Pipeline` to allow for simple access to the
    parameters involved in the definition of the Mapper algorithm, without the
    need to interface with the nested structure of the Pipeline objects
    generated by :func:`~gtda.mapper.pipeline.make_mapper_pipeline`. The
    convenience method :meth:`get_mapper_params` shows which parameters can
    be set. See the Examples below.

    Examples
    --------
    >>> from sklearn.cluster import DBSCAN
    >>> from sklearn.decomposition import PCA
    >>> from gtda.mapper import make_mapper_pipeline, CubicalCover
    >>> filter_func = PCA(n_components=2)
    >>> cover = CubicalCover()
    >>> clusterer = DBSCAN()
    >>> pipe = make_mapper_pipeline(filter_func=filter_func,
    ...                             cover=cover,
    ...                             clusterer=clusterer)
    >>> print(pipe.get_mapper_params()["clusterer__eps"])
    0.5
    >>> pipe.set_params(clusterer___eps=0.1)
    >>> print(pipe.get_mapper_params()["clusterer__eps"])
    0.1

    See also
    --------
    make_mapper_pipeline

    """

    # TODO: Abstract away common logic into a more generalisable implementation
    def get_mapper_params(self, deep=True):
        """Get all Mapper parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: ``True``
            If ``True``, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        pipeline_params = super().get_params(deep=deep)
        return {**{param: pipeline_params[param]
                   for param in global_pipeline_params},
                **self._clean_dict_keys(pipeline_params, nodes_params_prefix),
                **self._clean_dict_keys(
                    pipeline_params, clust_prepr_params_prefix),
                **self._clean_dict_keys(pipeline_params, clust_params_prefix),
                **self._clean_dict_keys(pipeline_params, nerve_params_prefix)}

    def set_params(self, **kwargs):
        """Set the Mapper parameters.

        Valid parameter keys can be listed with :meth:`get_mapper_params()`.

        Returns
        -------
        self

        """
        mapper_nodes_kwargs = self._subset_kwargs(kwargs, nodes_params)
        mapper_clust_prepr_kwargs = \
            self._subset_kwargs(kwargs, clust_prepr_params)
        mapper_clust_kwargs = self._subset_kwargs(kwargs, clust_params)
        mapper_nerve_kwargs = self._subset_kwargs(kwargs, nerve_params)
        if mapper_nodes_kwargs:
            super().set_params(
                **{nodes_params_prefix + key: mapper_nodes_kwargs[key]
                   for key in mapper_nodes_kwargs})
            [kwargs.pop(key) for key in mapper_nodes_kwargs]
        if mapper_clust_prepr_kwargs:
            super().set_params(
                **{clust_prepr_params_prefix + key:
                    mapper_clust_prepr_kwargs[key] for key in
                   mapper_clust_prepr_kwargs})
            [kwargs.pop(key) for key in mapper_clust_prepr_kwargs]
        if mapper_clust_kwargs:
            super().set_params(
                **{clust_params_prefix + key: mapper_clust_kwargs[key]
                   for key in mapper_clust_kwargs})
            [kwargs.pop(key) for key in mapper_clust_kwargs]
        if mapper_nerve_kwargs:
            super().set_params(
                **{nerve_params_prefix + key: mapper_nerve_kwargs[key]
                   for key in mapper_nerve_kwargs})
            [kwargs.pop(key) for key in mapper_nerve_kwargs]
        super().set_params(**kwargs)
        return self

    @staticmethod
    def _subset_kwargs(kwargs, param_strings):
        return {key: value for key, value in kwargs.items()
                if key.startswith(param_strings)}

    @staticmethod
    def _clean_dict_keys(kwargs, prefix):
        return {
            key[len(prefix):]: kwargs[key]
            for key in kwargs
            if (key.startswith(prefix)
                and not key.startswith(prefix + "steps")
                and not key.startswith(prefix + "memory")
                and not key.startswith(prefix + "verbose")
                and not key.startswith(prefix + "transformer_list")
                and not key.startswith(prefix + "n_jobs")
                and not key.startswith(prefix + "transformer_weights")
                and not key.startswith(prefix + "map_and_cover"))
            }


def make_mapper_pipeline(scaler=None,
                         filter_func=None,
                         cover=None,
                         clustering_preprocessing=None,
                         clusterer=None,
                         n_jobs=None,
                         parallel_backend_prefer=None,
                         graph_step=True,
                         min_intersection=1,
                         store_edge_elements=False,
                         contract_nodes=False,
                         memory=None,
                         verbose=False):
    """Construct a MapperPipeline object according to the specified Mapper
    steps [1]_.

    The role of this function's main parameters is illustrated in `this diagram
    <../../../../_images/mapper_pipeline.svg>`_. All computational steps may
    be scikit-learn estimators, including Pipeline objects.

    Parameters
    ----------
    scaler : object or None, optional, default: ``None``
        If ``None``, no scaling is performed. Otherwise, it must be an
        object with a ``fit_transform`` method.

    filter_func : object, callable or None, optional, default: ``None``
        If ``None``, PCA (:class:`sklearn.decomposition.PCA`) with 2
        components and default parameters is used as a default filter
        function. Otherwise, it may be an object with a ``fit_transform``
        method, or a callable acting on one-dimensional arrays -- in which
        case the callable is applied independently to each row of the
        (scaled) data.

    cover : object or None, optional, default: ``None``
        Covering transformer, e.g. an instance of
        :class:`~gtda.mapper.OneDimensionalCover` or of
        :class:`~gtda.mapper.CubicalCover`. ``None`` is equivalent to passing
        an instance of :class:`~gtda.mapper.CubicalCover` with its default
        parameters.

    clustering_preprocessing : object or None, optional, default: ``None``
        If not ``None``, it is a transformer which is applied to the
        data independently to the `scaler` -> `filter_func` -> `cover`
        pipeline. Clustering is then performed on portions (determined by
        the `scaler` -> `filter_func` -> `cover` pipeline) of the transformed
        data.

    clusterer : object or None, optional, default: ``None``
        Clustering object with a ``fit`` method which stores cluster labels.
        ``None`` is equivalent to passing an instance of
        :class:`sklearn.cluster.DBSCAN` with its default parameters.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use in a joblib-parallel application of the
        clustering step across pullback cover sets. To be used in
        conjunction with `parallel_backend_prefer`. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    parallel_backend_prefer : ``"processes"`` | ``"threads"`` | ``None``, \
        optional, default: ``None``
        Soft hint for the default joblib backend to use in a joblib-parallel
        application of the clustering step across pullback cover sets. To be
        used in conjunction with `n_jobs`. The default process-based backend is
        "loky" and the default thread-based backend is "threading". See [2]_.

    graph_step : bool, optional, default: ``True``
        Whether the resulting pipeline should stop at the calculation of the
        (refined) Mapper cover, or include the construction of the Mapper
        graph.

    min_intersection : int, optional, default: ``1``
        Minimum size of the intersection between clusters required for creating
        an edge in the Mapper graph. Ignored if `graph_step` is set to
        ``False``.

    store_edge_elements : bool, optional, default: ``False``
        Whether the indices of data elements associated to Mapper edges (i.e.
        in the intersections allowed by `min_intersection`) should be stored in
        the :class:`igraph.Graph` object output by the pipeline's
        :meth:`fit_transform`. When ``True``, might lead to large
        :class:`igraph.Graph` objects.

    contract_nodes : bool, optional, default: ``False``
        If ``True``, any node representing a cluster which is a strict subset
        of the cluster corresponding to another node is eliminated, and only
        one maximal node is kept.

    memory : None, str or object with the joblib.Memory interface, \
        optional, default: ``None``
        Used to cache the fitted transformers which make up the pipeline. This
        is advantageous when the fitting of early steps is time consuming and
        only later steps in the pipeline are modified (e.g. using
        :meth:`set_params`) before refitting on the same data. To be used
        exactly as for :func:`sklearn.pipeline.make_pipeline`. By default, no
        no caching is performed. If a string is given, it is the path to the
        caching directory. See [3]_.

    verbose : bool, optional, default: ``False``
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    mapper_pipeline : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Output Mapper pipeline. The output of `mapper_pipeline`'s
        :meth:`fit_transform` is: a) an :class:`igraph.Graph` object as per the
        output of :class:`~gtda.mapper.nerve.Nerve`, when `graph_step` is
        ``True``; b) a list of lists of tuples as per the output of
        :class:`~gtda.mapper.ParallelClustering` (or input of
        :class:`~gtda.mapper.Nerve`), otherwise.

    Examples
    --------
    Basic usage with default parameters

    >>> import numpy as np
    >>> from gtda.mapper import make_mapper_pipeline
    >>> mapper = make_mapper_pipeline()
    >>> print(mapper.__class__)
    <class 'gtda.mapper.pipeline.MapperPipeline'>
    >>> mapper_params = mapper.get_mapper_params()
    >>> print(mapper_params["filter_func"].__class__)
    <class 'sklearn.decomposition._pca.PCA'>
    >>> print(mapper_params["cover"].__class__)
    <class 'gtda.mapper.cover.CubicalCover'>
    >>> print(mapper_params["clusterer"].__class__)
    <class 'sklearn.cluster._dbscan.DBSCAN'>
    >>> X = np.random.random((10000, 4))  # 10000 points in 4-dimensional space
    >>> mapper_graph = mapper.fit_transform(X)  # Create the mapper graph
    >>> print(type(mapper_graph))
    igraph.Graph
    >>> # Node metadata stored as vertex attributes in graph object
    >>> print(mapper_graph.vs.attributes())
    ['pullback_set_label', 'partial_cluster_label', 'node_elements']
    >>> # Find which points belong to first node of graph
    >>> node_id = 0
    >>> node_elements = mapper_graph.vs["node_elements"]
    >>> print(f"Node ID: {node_id}, Node elements: {node_elements[node_id]}, "
    ...       f"Data points: {X[node_elements[node_id]")
    Node Id: 0,
    Node elements: [8768],
    Data points: [[0.01838998 0.76928754 0.98199244 0.0074299 ]]

    Using a scaler from scikit-learn, a filter function from
    ``gtda.mapper.filter``, and a clusterer from ``gtda.mapper.cluster``

    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from gtda.mapper import Projection, FirstHistogramGap
    >>> scaler = MinMaxScaler()
    >>> filter_func = Projection(columns=[0, 1])
    >>> clusterer = FirstHistogramGap()
    >>> mapper = make_mapper_pipeline(scaler=scaler,
    ...                               filter_func=filter_func,
    ...                               clusterer=clusterer)

    Using a callable acting on each row of X separately

    >>> import numpy as np
    >>> from gtda.mapper import OneDimensionalCover
    >>> cover = OneDimensionalCover()
    >>> mapper.set_params(scaler=None, filter_func=np.sum, cover=cover)

    Setting the memory parameter to cache each step and avoid recomputation
    of early steps

    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> cachedir = mkdtemp()
    >>> mapper.set_params(memory=cachedir, verbose=True)
    >>> mapper_graph = mapper.fit_transform(X)
    [Pipeline] ............ (step 1 of 3) Processing scaler, total=   0.0s
    [Pipeline] ....... (step 2 of 3) Processing filter_func, total=   0.0s
    [Pipeline] ............. (step 3 of 3) Processing cover, total=   0.0s
    [Pipeline] .... (step 1 of 3) Processing pullback_cover, total=   0.0s
    [Pipeline] ........ (step 2 of 3) Processing clustering, total=   0.3s
    [Pipeline] ............. (step 3 of 3) Processing nerve, total=   0.0s
    >>> mapper.set_params(min_intersection=3)
    >>> mapper_graph = mapper.fit_transform(X)
    [Pipeline] ............. (step 3 of 3) Processing nerve, total=   0.0s
    >>> # Clear the cache directory when you don't need it anymore
    >>> rmtree(cachedir)

    Using a large dataset for which parallelism in clustering across
    the pullback cover sets can be beneficial

    >>> from sklearn.cluster import DBSCAN
    >>> mapper = make_mapper_pipeline(clusterer=DBSCAN(),
    ...                               n_jobs=6,
    ...                               memory=mkdtemp(),
    ...                               verbose=True)
    >>> X = np.random.random((100000, 4))
    >>> mapper.fit_transform(X)
    [Pipeline] ............ (step 1 of 3) Processing scaler, total=   0.0s
    [Pipeline] ....... (step 2 of 3) Processing filter_func, total=   0.1s
    [Pipeline] ............. (step 3 of 3) Processing cover, total=   0.6s
    [Pipeline] .... (step 1 of 3) Processing pullback_cover, total=   0.7s
    [Pipeline] ........ (step 2 of 3) Processing clustering, total=   1.9s
    [Pipeline] ............. (step 3 of 3) Processing nerve, total=   0.3s
    >>> mapper.set_params(n_jobs=1)
    >>> mapper.fit_transform(X)
    [Pipeline] ........ (step 2 of 3) Processing clustering, total=   5.3s
    [Pipeline] ............. (step 3 of 3) Processing nerve, total=   0.3s

    See also
    --------
    MapperPipeline, method_to_transform

    References
    ----------
    .. [1] G. Singh, F. Mémoli, and G. Carlsson, "Topological methods for the
           analysis of high dimensional data sets and 3D object recognition";
           in *SPBG*, pp. 91--100, 2007.

    .. [2] "Thread-based parallelism vs process-based parallelism", in
           `joblib documentation
           <https://joblib.readthedocs.io/en/latest/parallel.html>`_.

    .. [3] "Caching transformers: avoid repeated computation", in
            `scikit-learn documentation \
            <https://scikit-learn.org/stable/modules/compose.html>`_.

    """

    # TODO: Implement parameter validation

    if scaler is None:
        _scaler = identity(validate=False)
    else:
        _scaler = scaler

    # If filter_func is not a scikit-learn transformer, hope it is a callable
    # to be applied on each row separately. Then attempt to create a
    # FunctionTransformer object to implement this behaviour.
    if filter_func is None:
        from sklearn.decomposition import PCA
        _filter_func = PCA(n_components=2)
    elif not hasattr(filter_func, "fit_transform"):
        _filter_func = transformer_from_callable_on_rows(filter_func)
    else:
        _filter_func = filter_func

    if cover is None:
        from .cover import CubicalCover
        _cover = CubicalCover()
    else:
        _cover = cover

    if clustering_preprocessing is None:
        _clustering_preprocessing = identity(validate=True)
    else:
        _clustering_preprocessing = clustering_preprocessing

    if clusterer is None:
        from sklearn.cluster import DBSCAN
        _clusterer = DBSCAN()
    else:
        _clusterer = clusterer

    map_and_cover = Pipeline(
        steps=[("scaler", _scaler),
               ("filter_func", _filter_func),
               ("cover", _cover)],
        verbose=verbose)

    all_steps = [
        ("pullback_cover", ListFeatureUnion(
            [("clustering_preprocessing", _clustering_preprocessing),
             ("map_and_cover", map_and_cover)])),
        ("clustering", ParallelClustering(
            _clusterer,
            n_jobs=n_jobs,
            parallel_backend_prefer=parallel_backend_prefer))
        ]

    if graph_step:
        all_steps.append(
            ("nerve", Nerve(min_intersection=min_intersection,
                            store_edge_elements=store_edge_elements,
                            contract_nodes=contract_nodes))
            )

    mapper_pipeline = MapperPipeline(
        steps=all_steps, memory=memory, verbose=verbose)

    return mapper_pipeline
