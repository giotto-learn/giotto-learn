"""Point-cloud–related plotting functions and classes."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj

from gtda.utils import validate_params
from gtda.utils.intervals import Interval as Int


def plot_point_cloud(point_cloud,
                     labels=None,
                     names=None,
                     dimension=None,
                     plotly_params=None,
                     marker_size=5,
                     opacity=0.8,
                     to_scale=False,
                     display_plot=False):
    """Plot the first 2 or 3 coordinates of a point cloud.

    Note: this function does not work on 1D arrays.

    Parameters
    ----------
    point_cloud : ndarray of shape (n_samples, n_dimensions)
        Data points to be represented in a 2D or 3D scatter plot. Only the
        first 2 or 3 dimensions will be considered for plotting.

    labels : ndarray of shape (n_samples,) or None, optional, default: ``None``
        Array of labels of data points that, if provided, are used to
        color-code the data points.

    names: dict or None, optional, default: ``None``
        Dictionary translating each numeric label into a string representing
        its name. Should be of the format {label[int] : name[str]}.
        If provided, a legend will be added to the plot.

    dimension : int or None, default: ``None``
        Sets the dimension of the resulting plot. If ``None``, the dimension
        will be chosen between 2 and 3 depending on the shape of `point_cloud`.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"trace"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    marker_size : float or None, optional, default: 5
        Sets the size of the markers in the plot. Must be a positive number.

    opacity : float or None, optional, default: 0.8
        Sets the opacity of the markers in the plot. Must be a number between
        0 and 1.

    to_scale : bool or None, optional, default: False
        Whether or not to use the same scale across all axes of the plot.

    show_plot: bool or None, optional, default: True
        Whether or not to display the plot.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing a point cloud in 2D or 3D.

    """

    # If no labels provided, just enumerate data points, and record
    # if there were user-provided labels to use in `names`.
    labels_were_provided = labels is not None
    if not labels_were_provided:
        labels = np.arange(point_cloud.shape[0])

    validate_params({"labels": labels},
                    {"labels": {"type": (np.ndarray,),
                                "of": {"type": (np.number,)}}})
    validate_params({"names": names},
                    {"names": {"type": (dict, type(None))}})
    validate_params({"dimension": dimension},
                    {"dimension": {"type": (int, type(None)),
                                   "in": [2, 3]}})
    validate_params({"marker_size": marker_size},
                    {"marker_size": {"type": (float, int),
                                     "in": Int(0, np.inf, closed="neither")}})
    validate_params({"opacity": opacity},
                    {"opacity": {"type": (float, int),
                                 "in": Int(0, 1, closed="right")}})
    validate_params({"to_scale": to_scale},
                    {"to_scale": {"type": (bool,)}})
    validate_params({"display_plot": display_plot},
                    {"display_plot": {"type": (bool,)}})

    if dimension is None:
        dimension = np.min((3, point_cloud.shape[1]))

    if names is not None:
        if not labels_were_provided:
            raise ValueError("No lables were provided.")
        all_labels_have_names = (
            np.array(
                [label in names.keys() for label in np.unique(labels)]
            ).all()
            )
        if not all_labels_have_names:
            raise ValueError(
                "One or more labels are lacking a corresponding name."
                )
        all_names_are_strings = (
            np.array(
                [type(value) == str for value in names.values()]
                ).all()
        )
        if all_names_are_strings:
            raise TypeError(
                "All values of `names` should be strings."
                )

    # Check consistency between point_cloud and dimension
    if point_cloud.shape[1] < dimension:
        raise ValueError("Not enough dimensions available in the input point "
                         "cloud.")

    elif dimension == 2:
        layout = {
            "width": 600,
            "height": 600,
            "xaxis1": {
                "title": "0th",
                "side": "bottom",
                "type": "linear",
                "ticks": "outside",
                "anchor": "x1",
                "showline": True,
                "zeroline": True,
                "showexponent": "all",
                "exponentformat": "e"
                },
            "yaxis1": {
                "title": "1st",
                "side": "left",
                "type": "linear",
                "ticks": "outside",
                "anchor": "y1",
                "showline": True,
                "zeroline": True,
                "showexponent": "all",
                "exponentformat": "e"
                },
            "plot_bgcolor": "white"
            }

        fig = gobj.Figure(layout=layout)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor="black",
                         mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor="black",
                         mirror=False)

        if names is None:
            fig.add_trace(gobj.Scatter(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                mode="markers",
                marker={"size": marker_size,
                        "color": labels,
                        "colorscale": "Viridis",
                        "opacity": opacity}
                ))
        else:
            for label in np.unique(labels):
                fig.add_trace(gobj.Scatter(
                    x=point_cloud[labels == label][:, 0],
                    y=point_cloud[labels == label][:, 1],
                    mode="markers",
                    name=names[label],
                    marker={"size": marker_size,
                            "color": label,
                            "colorscale": "Viridis",
                            "opacity": opacity}
                ))
        if to_scale:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

    elif dimension == 3:
        scene = {
            "xaxis": {
                "title": "0th",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
                },
            "yaxis": {
                "title": "1st",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
                },
            "zaxis": {
                "title": "2nd",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
                }
            }

        fig = gobj.Figure()
        fig.update_layout(scene=scene)

        if names is None:
            fig.add_trace(gobj.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode="markers",
                marker={"size": marker_size,
                        "color": labels,
                        "colorscale": "Viridis",
                        "opacity": opacity}
                ))
        else:
            for label in np.unique(labels):
                fig.add_trace(gobj.Scatter3d(
                    x=point_cloud[labels == label][:, 0],
                    y=point_cloud[labels == label][:, 1],
                    z=point_cloud[labels == label][:, 2],
                    mode="markers",
                    name=names[label],
                    marker={"size": marker_size,
                            "color": label,
                            "colorscale": "Viridis",
                            "opacity": opacity}
                    ))
        if to_scale:
            fig.update_layout(scene_aspectmode='data')

    # Update trace and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("trace", None))
        fig.update_layout(plotly_params.get("layout", None))

    if display_plot:
        fig.show()
    return fig
