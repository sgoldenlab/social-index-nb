from io import BytesIO
from base64 import b64encode
from click import group
import matplotlib.pyplot as plt
import altair as alt
from typing import Union, Tuple, Any
from scipy.stats import norm as normf
from sklearn import mixture
from oss_app.dataset import Dataset
from scipy.stats import ks_2samp
from sklearn import decomposition
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import pandas as pd
import marimo as mo
import numpy as np

from oss_app.utils import make_categorical, mo_print


class ColorSet:
    """A class to handle color mapping for datasets."""

    def __init__(self,
                 color_name: str, metric_name: str,
                 grouping_variable: str, group_name: str, data: pd.DataFrame | pd.Series | np.ndarray | None = None,
                 cmap_range: tuple[float, float] = (0.4, 1), n_divisions: int = 151
                 ):
        self.name = color_name
        self.metric_name = metric_name
        self.grouping_variable = grouping_variable
        if not isinstance(group_name, str):
            raise ValueError("group_name must be a string.")
        self.group_name = group_name
        self.data = data.copy()
        self.cmap_range = cmap_range if cmap_range else (0, 1)
        # get colormap based on name and range given
        self._initialize(self.data, color_name, n_divisions)

    def _initialize(self, data: pd.DataFrame | pd.Series | np.ndarray | None, color_name: str, n_divisions: int = 151):
        self.cmap = cm.get_cmap(color_name)
        self.divcmap = ListedColormap(self.cmap(np.linspace(
            self.cmap_range[0], self.cmap_range[1], n_divisions)), name=f"{color_name}_colorset")
        self.colors, self.hex_colors = self._generate_colors(data)

    def _generate_colors(self, data: pd.DataFrame | pd.Series | np.ndarray | None = None, metric_name: str | None = None):
        """Generates colors based on the colormap and data."""
        if data is None:
            data = self.data  # use the original data if not provided
        if metric_name is None:
            metric_name = self.metric_name  # default to instance variable

        if isinstance(data, pd.DataFrame):
            # values = data.select_dtypes(
            #     include=[np.number]).values.flatten()
            values = data[metric_name].values.flatten()
        elif isinstance(self.data, pd.Series):
            values = self.data.values.flatten()
        elif isinstance(self.data, np.ndarray):
            values = self.data.flatten()
        else:
            raise ValueError("Data must be a DataFrame or NumPy array.")

        # normed_values = (values - np.min(values)) / \
        #     (np.max(values) - np.min(values))
        self.vmin, self.vmax = np.min(values), np.max(values)
        norm = colors.Normalize(
            vmin=self.vmin, vmax=self.vmax)
        sm = plt.cm.ScalarMappable(cmap=self.divcmap, norm=norm)

        rgba_colors = sm.to_rgba(values)  # Get RGBA colors
        hex_colors = [colors.rgb2hex(v)
                      for v in rgba_colors]  # Convert RGB to hex
        return rgba_colors, hex_colors

    def _filter_data(self, data: pd.DataFrame | pd.Series | np.ndarray | None = None, grouping_variable: str | None = None):
        """Filters the data to the relevant group.
        This method updates the `filtered_data` attribute with the data for the specified group.
        If `data` is not provided, it uses the original data stored in the instance.<br>
        NOTE: `Colorset.colors` will now be based on the filtered data, run `_initialize(self.data)` to get original colors.
        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | None): The data to filter.
            grouping_variable (str | None): The grouping variable to filter by.
        Returns:
            pd.DataFrame | pd.Series | np.ndarray | None: The filtered data.
        """
        if data is None:
            data = self.data
        if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError(
                "Data must be a DataFrame, Series, or NumPy array.")
        if not grouping_variable and self.grouping_variable:
            grouping_variable = self.grouping_variable
        elif not grouping_variable:
            raise ValueError(
                "grouping_variable must be specified if not provided in the data.")

        data = data[data[self.grouping_variable] == self.group_name][self.metric_name] if isinstance(
            data, pd.DataFrame) else data[data[self.grouping_variable] == self.group_name]

        self.filtered_data = data
        # regenerate colors based on filtered data
        #   NOTE: colors are now based on the filtered data
        self.colors = self._generate_colors(self.filtered_data)
        return self.filtered_data

    def _map_colors(self, data: pd.DataFrame | np.ndarray | None = None):
        """Maps colors to groups based on their values for given variable.
        By default the variable is `si_score` for their index scores."""
        if data is None:
            data = self.data
        if isinstance(data, pd.DataFrame):
            return data.apply(lambda x: self.colors[x.name], axis=1)
        elif isinstance(data, np.ndarray):
            return np.array([self.colors[i] for i in range(len(data))])
        else:
            raise ValueError("Data must be a DataFrame or NumPy array.")

    def blend_colors(self, other_color_set: 'ColorSet', split: bool = True):
        """Blends this color set with another ColorSet."""
        if not isinstance(other_color_set, ColorSet):
            raise ValueError(
                "other_color_set must be an instance of ColorSet.")

        top_half = cm.get_cmap(self.name, 1024)
        bottom_half = cm.get_cmap(other_color_set.name, 1024)

        if split:
            new_colors = np.vstack(
                (top_half(np.linspace(0, 0.5, 1024)), bottom_half(np.linspace(0.5, 1, 1024))))
        else:
            new_colors = np.vstack(
                (
                    top_half(np.linspace(
                        1 - self.cmap_range[1], 1 - self.cmap_range[0], 1024)),
                    bottom_half(np.linspace(
                        self.cmap_range[0], self.cmap_range[1], 1024)),
                )
            )
        return ListedColormap(new_colors, name=f"blended_{self.name}_{other_color_set.name}")

    def __repr__(self):
        return f"ColorSet(name={self.name}, group_name={self.group_name}, cmap_range={self.cmap_range})"


def show_colormap(cmap: Union[str, ListedColormap], label: str = "", figsize: Tuple[float, float] = (1.25, 0.25), font_family: str = "arial"):
    """Displays a visual representation of a Matplotlib colormap within a Marimo cell.
    This function generates a horizontal color bar for the given colormap and
    embeds it as an HTML object in a Marimo markdown cell, making it easy to
    visualize colormaps directly in a notebook.
        cmap: The Matplotlib colormap to display. This can be the string name
            of a registered colormap (e.g., "viridis") or a `matplotlib.colors.Colormap`
            object.
        label: An optional text label to display next to the colormap visualization.
        figsize: A tuple specifying the (width, height) in inches for the
            generated color bar figure.
    Returns:
        A `marimo.md` object containing the HTML representation of the colormap
        visualization, ready to be displayed in a Marimo cell.
    """
    fig, ax = plt.subplots(figsize=figsize, frameon=False,
                           facecolor='white')
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.set_axis_off()
    ax.set_xmargin(0)

    # use a BytesIO object as an in-memory buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close(fig)  # Close the figure to free memory

    # rewind the buffer
    buf.seek(0)

    image_element = mo.image(
        buf.getvalue(),
        alt="colormap",
    )

    if label != "":
        # Create a marimo markdown object for the label
        label_md = mo.md(
            f"<span style='font-family:{font_family}'>{label}</span>")
        # Use hstack to place them side by side
        return mo.hstack([label_md, image_element], justify='start', align='start')
    else:
        return mo.hstack([image_element], justify='start', align='start')


def show_color(color: str, label: str = "", font_family: str = "arial") -> mo.md:
    """Generates a Marimo markdown object to display a color swatch.

    This function creates a small, colored square using HTML and CSS,
    wrapped in a Marimo markdown object, making it easy to visualize
    colors directly within a Marimo notebook.

    Args:
        color (str): The name or hexadecimal color code to display (e.g., "#FF0000").
        label (str, optional): A text label to show next to the color swatch.
            Defaults to an empty string.

    Returns:
        marimo.md: A Marimo markdown object that renders as a label
            followed by a colored square.
    """
    "example hex_code=#FF0000"
    if label != "":
        md_content = mo.md(
            f"""<span style='font-family:{font_family}'>{label}</span><span style="background-color:{color}; display: inline-block; width: 20px; height: 20px; border: 1px solid black;vertical-align:middle"></span>"""
        )
    else:
        md_content = mo.md(
            f"""<span style="background-color:{color}; display: inline-block; width: 20px; height: 20px; border: 1px solid black;vertical-align:top"></span>"""
        )
    return md_content.style({'font-family': 'sans-serif'})


def colormap_to_hex(cmap: Union[str, ListedColormap], num_colors: int = 10) -> list[str]:
    """Converts a Matplotlib colormap to a list of hexadecimal color codes.

    Args:
        cmap (Union[str, ListedColormap]): The colormap to convert. This can be
            the name of a registered colormap (e.g., "viridis") or a
            `matplotlib.colors.Colormap` object.
        num_colors (int): The number of colors to extract from the colormap.
            Defaults to 10.

    Returns:
        list[str]: A list of hexadecimal color codes representing the
            specified number of colors from the colormap.
    """
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    return [colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, num_colors)]


# %% Plotting functions
# Distribution comparison
def _split_data_by_group(
    df_to_use: pd.DataFrame, compvar: str, groupvar: str, filters: dict
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Splits filtered dataframe into two arrays based on a grouping variable.
    """

    assert isinstance(
        df_to_use, pd.DataFrame), "df_to_use must be a pandas DataFrame"
    if filters:
        query_str = " & ".join(
            [f'`{var}`=="{val}"' if isinstance(
                val, str) else f"`{var}`=={val}" for var, val in filters.items()]
        )
        df_to_use = df_to_use.query(query_str)

    comp_grps = df_to_use.groupby(by=groupvar)

    if len(comp_grps) != 2:
        raise ValueError(
            f"Expected 2 groups after filtering, but found {len(comp_grps)}. Please filter data to two groups."
        )

    comp_arr_labels = list(comp_grps.groups.keys())
    comp_arrays = comp_grps[compvar].apply(
        lambda x: x.to_numpy() if isinstance(x, pd.Series) else np.array(x)).values

    return comp_arrays[0], comp_arrays[1], comp_arr_labels


def _calculate_distribution(data_array: np.ndarray, f_axis: np.ndarray) -> tuple[np.ndarray, dict]:
    # Fit a Gaussian Mixture Model (MLE for single Gaussian)
    f = np.ravel(data_array).astype(np.float64).reshape(-1, 1)
    g = mixture.GaussianMixture(
        n_components=1, covariance_type="full", random_state=0)
    g.fit(f)
    weights, means, covars = g.weights_, g.means_, g.covariances_

    mean = means[0, 0]
    std_dev = np.sqrt(covars[0, 0, 0])

    pdf = normf.pdf(f_axis, loc=mean, scale=std_dev).ravel()

    # This will make the curve integrate to 100 over its range
    dist_freq = pdf * 100  # scale for percentage density

    gmm_params = {"mean": mean, "std": std_dev}
    return dist_freq, gmm_params


def _calculate_intersection_and_overlap(gmm_params1: dict, gmm_params2: dict, x_range: list | np.ndarray):
    """
    Calculate overlap coefficient (using the GMM-derived parameters)
    Uses the raw norm.pdf for the coefficient, then multiplies by 100 for percentage
    """
    # using estimated loc and scale from GMM parameters
    mean1_gmm_est, std1_gmm_est = gmm_params1["mean"], gmm_params1["std"]
    mean2_gmm_est, std2_gmm_est = gmm_params2["mean"], gmm_params2["std"]

    pdf_group1_normalized_for_overlap = normf.pdf(
        x_range, loc=mean1_gmm_est, scale=std1_gmm_est)
    pdf_group2_normalized_for_overlap = normf.pdf(
        x_range, loc=mean2_gmm_est, scale=std2_gmm_est)

    min_of_normalized_pdfs = np.minimum(
        pdf_group1_normalized_for_overlap, pdf_group2_normalized_for_overlap)
    dx_plot = x_range[1] - x_range[0]
    # This is the [0,1] coefficient
    overlap_coefficient = np.sum(min_of_normalized_pdfs) * dx_plot
    overlap_percentage_value = overlap_coefficient * 100

    print(f"\nCalculated Overlap Coefficient: {overlap_coefficient:.4f}")
    print(f"Calculated Overlap Percentage: {overlap_percentage_value:.2f}%")
    return overlap_coefficient, overlap_percentage_value


def _perform_ks_test(data_array1: np.ndarray, data_array2: np.ndarray) -> tuple[float, float]:
    """
    Performs and reports the Kolmogorov-Smirnov test between two groups.
    """
    ks_stat, ks_pval = ks_2samp(
        data_array1.reshape(-1), data_array2.reshape(-1))
    print(
        f"\t--Kolmogorov-Smirnov test between groups: {ks_stat:.2f},  p-val: {ks_pval:.5f}")
    return ks_stat, ks_pval


def compare_dists_altair(
    df: pd.DataFrame,
    compare_metric="",
    group_variable="",
    colorset: ColorSet | None = None,
    filters={},
    max_y=40,
    rangex: list | None = [-2.5, 2.5],
    bin_width=1,
    dot=False,
    user_colors=None,
    user_labels: list[str, str] = None,
    set_size_params=None,
    legend=False,
    hide_text=False,
    alpha1=0.7,
    alpha2=0.4,
):
    """
    Compares distribution between two groups for a given variable using Altair.
    This function is a wrapper that utilizes helper methods to perform its core tasks:
    1. _split_data_by_group: Splits data into two arrays for comparison.
    2. _calculate_distribution: Calculates the GMM distribution for each group.
    3. _calculate_intersection_and_overlap: Finds the intersection and overlap coefficient.
    4. _perform_ks_test: Runs a Kolmogorov-Smirnov test.
    The final output is an Altair chart object.

    :param compvar: Comparison variable name.
    :type compvar: str
    :param groupvar: Grouping variable name.
    :type groupvar: str
    :param filters: Dictionary of filters to apply to the dataframe.
    :type filters: dict
    :param max_y: Maximum y-axis value.
    :type max_y: int
    :param rangex: Range for the x-axis, e.g., [-2.5, 2.5].
    :type rangex: list
    :param binw: Bin width for density calculation scaling.
    :type binw: int
    :param dot: Whether to plot the intersection point.
    :type dot: bool
    :param user_colors: Tuple of two colors for the groups.
    :type user_colors: tuple[str, str], optional
    :param user_labels: List of two labels for the groups.
    :type user_labels: list[str, str], optional
    :param set_size_params: Tuple (width, height) in inches for the plot size.
    :type set_size_params: tuple[float, float], optional
    :param legend: Whether to display the legend.
    :type legend: bool
    :param hide_text: Whether to hide all text (titles, labels, ticks).
    :type hide_text: bool
    :param alpha1: Opacity for the first group's area.
    :type alpha1: float
    :param alpha2: Opacity for the second group's area.
    :type alpha2: float
    :return: An Altair chart object.
    :rtype: alt.Chart
    """
    assert compare_metric != "", 'no comparison variable for "compare_metric" chosen'
    assert group_variable != "", 'no grouping variable for "group_variable" chosen'

    # 1. Split data into two groups
    array1, array2, comp_arr_labels = _split_data_by_group(
        df, compare_metric, group_variable, filters)

    plot_title = f"{compare_metric=},  {group_variable=}"

    # 2. Calculate distributions for each group
    concat_min = np.concatenate((array1, array2)).min()
    concat_max = np.concatenate((array1, array2)).max()
    if not rangex:  # If rangex is not provided, set it based on data but wider
        rangex = [round(concat_min - 5), round(concat_max + 5)]
    range_min, range_max = rangex
    smooth_factor = 500
    x_plot_range = np.linspace(
        # Increase smooth_factor for smoother curves
        range_min,
        range_max,
        smooth_factor,
    )

    # Calculate distributions and get GMM parameters
    dist_freq_group1, gmm_params_group1 = _calculate_distribution(
        data_array=array1,
        f_axis=x_plot_range,
    )
    mean1_gmm_est = gmm_params_group1["mean"]
    std1_gmm_est = gmm_params_group1["std"]

    dist_freq_group2, gmm_params_group2 = _calculate_distribution(
        data_array=array2,
        f_axis=x_plot_range,
    )
    mean2_gmm_est = gmm_params_group2["mean"]
    std2_gmm_est = gmm_params_group2["std"]

    print(
        f"GMM Group 1 Estimates: Mean={mean1_gmm_est:.2f}, Std={std1_gmm_est:.2f}")
    print(
        f"GMM Group 2 Estimates: Mean={mean2_gmm_est:.2f}, Std={std2_gmm_est:.2f}")

    # 3. Calculate intersection and overlap coefficient
    overlap_coeff, overlap_pct = _calculate_intersection_and_overlap(
        gmm_params_group1, gmm_params_group2, x_plot_range)

    # 4. Perform and report Kolmogorov-Smirnov test
    ks_stat, ks_pval = _perform_ks_test(array1, array2)

    # --- Altair Charting ---
    # Prepare data for Altair DataFrame

    _user_colors = user_colors if user_colors else ("#e45756", "#4c78a8")
    _lgnd_labels = user_labels if user_labels is not None else [
        str(lbl) for lbl in comp_arr_labels]

    data_dist_list = []
    for val_x, val_y in zip(x_plot_range, dist_freq_group1):
        data_dist_list.append(
            {"f_axis": val_x, "dist_val": val_y, "group": _lgnd_labels[0], "alpha_val": alpha1})
    for val_x, val_y in zip(x_plot_range, dist_freq_group2):
        data_dist_list.append(
            {"f_axis": val_x, "dist_val": val_y, "group": _lgnd_labels[1], "alpha_val": alpha2})
    source_dist = pd.DataFrame(data_dist_list)

    # Base chart properties
    plot_width = set_size_params[0] * 96 if set_size_params else 192
    plot_height = set_size_params[1] * 96 if set_size_params else 192
    if not max_y:
        concat_max = max(dist_freq_group1.max(), dist_freq_group2.max())
        max_y = round(concat_max + 10)
    # Create Altair chart
    area_chart = (
        alt.Chart(source_dist)
        .mark_area(
            line={"color": "black", "strokeWidth": 1.5, "strokeOpacity": 1},
            strokeWidth=0,
        )
        .encode(
            x=alt.X("f_axis:Q", title=f"Var: {compare_metric}", scale=alt.Scale(
                domain=rangex, nice=False, zero=False)),
            y=alt.Y(
                "dist_val:Q",
                title="% Subjects",
                stack=False,
                scale=alt.Scale(domain=[0, max_y], nice=False, zero=True),
                axis=alt.Axis(format=".0f"),
            ),
            fill=alt.Fill(
                "group:N",
                scale=alt.Scale(domain=_lgnd_labels, range=_user_colors),
                legend=alt.Legend(title=group_variable) if legend else None,
            ),
            fillOpacity=alt.FillOpacity("alpha_val:Q", legend=None),
        )
    )

    chart = area_chart

    # Titles and styling
    chart = chart.properties(
        width=plot_width,
        height=plot_height,
        title=alt.TitleParams(
            text=f"{plot_title}",
            subtitle=f"Overlap coeff: {overlap_coeff:.3f},  pct: {overlap_pct:.2f}%  |  K-S test: {ks_stat:.2f}, p-val: {ks_pval:.3f}",
            fontSize=12,
            subtitleFontSize=10,
            subtitleFontStyle="italic",
            anchor="middle",
            offset=10,
        ),
    )

    if hide_text:
        chart = chart.configure_axis(labels=False, title=None, ticks=False, grid=False).properties(
            title=alt.TitleParams(text="", subtitle="")
        )
    else:
        chart = chart

    return chart

# PCA


def do_pca(data: pd.DataFrame, n_comp=3):
    """Performs Principal Component Analysis (PCA) on the provided dataset.

    This function wraps `sklearn.decomposition.PCA` to simplify the process of
    dimensionality reduction. It fits the PCA model to the data, transforms the
    data into the principal component space, and returns the fitted model along
    with the results.

    Args:
        data (pd.DataFrame): A DataFrame containing the scaled numerical data
            (features) to be analyzed. The shape should be (n_samples, n_features).
        n_comp (int, float, optional): The number of principal components to keep.
            - If an integer, it's the absolute number of components.
            - If a float between 0.0 and 1.0 (e.g., 0.95), it's the amount of
              variance that should be explained by the selected components.
            Defaults to 3.

    Returns:
        tuple: A tuple containing the following three elements:
            - pca (sklearn.decomposition.PCA): The fitted PCA object from scikit-learn.
              This object can be used for further analysis or inverse transformations.
            - principal_components (np.ndarray): An array of shape
              (n_samples, n_components) representing the data transformed into the
              principal component space.
            - pc_evr (np.ndarray): An array containing the percentage of variance
              explained by each of the selected components.
    """
    pca = decomposition.PCA(n_components=n_comp)
    principal_components = pca.fit_transform(data)
    pc_evr = pca.explained_variance_ratio_
    return pca, principal_components, pc_evr


def pca_biplot_old(
    dataset_obj,
    df_to_use=pd.DataFrame() | str | None,
    grouping_variable="",
    metrics_inluded: list[str] | None = None,
    labels="",
    pca=None,
    n_comp=3,
    mapping=0,
    pcs=None,
    colmap=None,
    hide_text=False,
    **scatter_kwargs,
):
    df: pd.DataFrame = pd.DataFrame()

    # Data prep
    assert not df_to_use.empty, "Dataset object's input DataFrame `df_to_use` cannot be empty."
    if isinstance(df_to_use, str):
        assert getattr(dataset_obj, df_to_use,
                       None) is not None, f"Dataset object has no dataset `{df_to_use}`."
        df = getattr(dataset_obj, df_to_use)
    elif df_to_use is None:
        assert getattr(dataset_obj, "scaled_df",
                       None) is not None, "Default dataset `scaled_df` not found."
        df = getattr(dataset_obj, "scaled_df")
    else:
        df = df_to_use
    if not metrics_inluded:
        # metrics_inluded = [col for col in dataset_obj.metric_variables if col != "si_score"]
        metrics_inluded = dataset_obj.metric_variables
    metric_labels = metrics_inluded

    assert not (
        df.empty or df is None), "Dataset object's input DataFrame `df` cannot be None."
    # type: ignore # copy to avoid modifying original DataFrame
    df = df[metrics_inluded].copy()

    if not grouping_variable:
        grouping_variable = dataset_obj.grouping_variable
    group_categories = make_categorical(df[grouping_variable])

    # PCA
    pca, princomps, pc_evr = do_pca(df, n_comp)
    if not getattr(dataset_obj, "pca", None):
        dataset_obj.pca = do_pca(df, n_comp)
    if pcs is None:
        pc_range = range(n_comp)
        pcset_to_plot = [[x, y] for x, y in zip(pc_range[0:-1], pc_range[1:])]
    else:
        pcset_to_plot = [pcs]

    # Plotting setup
    x_index = group_categories["codes"]
    stylemap = {0: "v", 1: "o"}  # marker style for groups
    sizemap = {0: 70, 1: 60}  # marker size for groups
    conds_stylemap = [*map(stylemap.get, x_index)]
    conds_sizemap = [*map(sizemap.get, x_index)]
    if colmap is None:
        match not dataset_obj.plot_colors:
            case False:
                colmap = dataset_obj.plot_colors[0].new_cmap
            case _:
                dataset_obj.map_colors(dataset_obj.si_scores)
                colmap = dataset_obj.plot_colors[0].new_cmap
    mapping = dataset_obj.plot_colors[0].face_colors

    kwargs = dict(  # scatter parameters
        edgecolors="face",
        lw=0.5,
        zorder=2,
    )
    kwargs.update(scatter_kwargs)  # update kwargs with user parameters

    # set up legend entries
    leg_markers = [*map(stylemap.get, np.unique(group_categories["codes"]))]
    legend_handles = [
        Line2D([], [], marker=m, markeredgecolor="k",
               markeredgewidth=1.5, markerfacecolor="w", linewidth=0)
        for m in leg_markers
    ]

    # Plotting, per set of PCs given, only one if `pcs` not None
    for pcset in pcset_to_plot:
        coeff = np.transpose(pca.components_[pcset[0]: pcset[1] + 1])
        xs = princomps[:, pcset[0]]
        ys = princomps[:, pcset[1]]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        # pdb.set_trace()
        ncmap = colmap

        plt.figure(figsize=(3, 3))
        # plt.scatter(xs * scalex, ys * scaley, c=mapping, cmap=ncmap, s=25, edgecolors='face',
        #             marker=conds_stylemap, size=conds_sizemap, zorder=2)
        for _st, _si, _c, _x, _y in zip(conds_stylemap, conds_sizemap, mapping, xs, ys):
            fs = plt.scatter(_x * scalex, _y * scaley, color=_c,
                             cmap=ncmap, s=_si, marker=_st, **kwargs)
        for i in range(n):
            if not hide_text:
                match labels:
                    case "labeled":
                        # plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, f'{var_labels[i]}',
                        plt.text(
                            coeff[i, 0] + 0.1,
                            coeff[i, 1],
                            f"{metric_labels[i]}",
                            color="darkgreen",
                            ha="left",
                            va="center",
                        )
                    case "ordered":
                        plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.0,
                                 str(i), color="k", ha="center", va="center")
                    case "":
                        pass
            # arc_direction_list = iter(["-", "", "-", "-", "-"])
            plt.annotate(
                "",
                xy=(coeff[i, 0], coeff[i, 1]),
                xytext=(0, 0),  # color='crimson'
                arrowprops=dict(
                    arrowstyle="-|>",
                    shrinkA=0,
                    fill=True,
                    color="xkcd:coral",
                    mutation_scale=12,
                    lw=1.5,
                    #  connectionstyle= f'arc3,rad={next(arc_direction_list)}0.1',
                ),
                zorder=3,
            )

        plt.grid(zorder=1)
        plt.xlim(-1.2, 1.2)
        plt.xticks([-1, 0, 1])
        plt.ylim(-1.2, 1.2)
        plt.yticks([-1, 0, 1])
        # plt.xlim(lims:=(-.9, .9))
        # plt.xlim(lims:=(-.92, .92))
        # plt.xticks(ticklims:=[-0.5, 0, 0.5])
        # plt.ylim(lims)
        # plt.yticks(ticklims)
        plt.xlabel(f"PC{pcset[0] + 1}")
        plt.ylabel(f"PC{pcset[1] + 1}")
        plt.title(
            f"Explained variance ratio: \nPC{pcset[0] + 1}: {pc_evr[pcset[0]]:.2f},  PC{pcset[1] + 1}: {pc_evr[pcset[1]]:.2f} \nCombined:{sum(pc_evr[pcset]):.2f}"
        )
        if hide_text:
            # Hide X and Y axes label marks
            ax = plt.gca()
            plt.xlim(lims := (-0.92, 0.92))
            plt.ylim(lims)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            plt.xlabel(None)
            plt.ylabel(None)
            plt.title(None)
            plt.grid(visible=False)
            # single grid line instead
            ax.axhline(0, linestyle=":", color="xkcd:gray",
                       zorder=1)  # horizontal lines
            ax.axvline(0, linestyle=":", color="xkcd:gray",
                       zorder=1)  # vertical lines
        else:
            labels = list(group_categories["labels"])
            handles = legend_handles
            ax = plt.gca()
            ax.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=8,
                bbox_to_anchor=(1.5, 1),
                frameon=False,
                markerscale=1.5,
            )


def pca_biplot(
    df_to_use=pd.DataFrame(),
    grouping_variable="",
    metrics_included: list[str] | None = None,
    labels="",
    pca_inputs: list[Union[PCA, np.ndarray, ...]] = None,
    n_comp=3,
    mapping=0,
    pcs=None,
    colorset: ColorSet | None = None,
    hide_text=False,
    **scatter_kwargs,
):
    # data prep

    assert not df_to_use.empty, "Input DataFrame `df_to_use` cannot be empty."
    df = df_to_use.copy()
    if not metrics_included:
        metrics_included = df.select_dtypes(
            include=[np.number]).columns.tolist()
    metric_labels = metrics_included

    # PCA
    if not pca_inputs:
        pca, princomps, pc_evr = do_pca(df, n_comp)
    else:
        pca, princomps, pc_evr = pca_inputs
    if pcs is None:  # TODO: rename pcs and pcset_to_plot
        pc_range = range(n_comp)
        pcset_to_plot = [[x, y] for x, y in zip(pc_range[0:-1], pc_range[1:])]
    else:
        pcset_to_plot = [pcs]

    # Plotting setup
    assert grouping_variable != "", "Grouping variable must be specified."
    group_categories = make_categorical(df[grouping_variable])
    x_index = group_categories["codes"]
    style_mapping = {0: "v", 1: "o"}  # marker style for groups
    size_mapping = {0: 70, 1: 60}  # marker size for groups
    conds_stylemap = [style_mapping.get(i)
                      for i in x_index]  # TODO: rename conds_*
    conds_sizemap = [size_mapping.get(i) for i in x_index]
    if colorset is None:
        colorset = ColorSet(
            color_name="viridis_r",
            metric_name='si_score',  # choose a
            grouping_variable=grouping_variable,
            group_name=group_categories["labels"][0],
            data=df
        )
    mapping = colorset.colors

    kwargs = dict(  # scatter parameters
        edgecolors="face",
        lw=0.5,
        zorder=2,
    )
    kwargs.update(scatter_kwargs)  # update kwargs with user parameters

    # set up legend entries
    leg_markers = [
        *map(style_mapping.get, np.unique(group_categories["codes"]))]
    legend_handles = [
        Line2D([], [], marker=m, markeredgecolor="k",
               markeredgewidth=1.5, markerfacecolor="w", linewidth=0)
        for m in leg_markers
    ]

    # Plotting, per set of PCs given, only one if `pcs` not None
    for pcset in pcset_to_plot:
        coeff = np.transpose(pca.components_[pcset[0]: pcset[1] + 1])
        xs = princomps[:, pcset[0]]
        ys = princomps[:, pcset[1]]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        plt.figure(figsize=(3, 3))
        for _st, _si, _c, _x, _y in zip(conds_stylemap, conds_sizemap, mapping, xs, ys):
            fs = plt.scatter(_x * scalex, _y * scaley, color=_c,  # cmap=ncmap,
                             s=_si, marker=_st, **kwargs)

        for i in range(n):
            if not hide_text:
                match labels:
                    case "labeled":
                        # plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, f'{var_labels[i]}',
                        plt.text(
                            coeff[i, 0] + 0.1,
                            coeff[i, 1],
                            f"{metric_labels[i]}",
                            color="darkgreen",
                            ha="left",
                            va="center",
                        )
                    case "ordered":
                        plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.0,
                                 str(i), color="k", ha="center", va="center")
                    case "":
                        pass

            plt.annotate(
                "",
                xy=(coeff[i, 0], coeff[i, 1]),
                xytext=(0, 0),  # color='crimson'
                arrowprops=dict(
                    arrowstyle="-|>",
                    shrinkA=0,
                    fill=True,
                    color="xkcd:coral",
                    mutation_scale=12,
                    lw=1.5,
                    #  connectionstyle= f'arc3,rad={next(arc_direction_list)}0.1',
                ),
                zorder=3,
            )

            plt.grid(zorder=1)
            plt.xlim(-1.2, 1.2)
            plt.xticks([-1, 0, 1])
            plt.ylim(-1.2, 1.2)
            plt.yticks([-1, 0, 1])
            # plt.xlim(lims:=(-.9, .9))
            # plt.xlim(lims:=(-.92, .92))
            # plt.xticks(ticklims:=[-0.5, 0, 0.5])
            # plt.ylim(lims)
            # plt.yticks(ticklims)
            plt.xlabel(f"PC{pcset[0] + 1}")
            plt.ylabel(f"PC{pcset[1] + 1}")
            plt.title(
                f"Explained variance ratio: \nPC{pcset[0] + 1}: {pc_evr[pcset[0]]:.2f},  PC{pcset[1] + 1}: {pc_evr[pcset[1]]:.2f} \nCombined:{sum(pc_evr[pcset]):.2f}"
            )

            if hide_text:
                # Hide X and Y axes label marks
                ax = plt.gca()
                plt.xlim(lims := (-0.92, 0.92))
                plt.ylim(lims)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                plt.xlabel(None)
                plt.ylabel(None)
                plt.title(None)
                plt.grid(visible=False)
                # single grid line instead
                ax.axhline(0, linestyle=":", color="xkcd:gray",
                           zorder=1)  # horizontal lines
                ax.axvline(0, linestyle=":", color="xkcd:gray",
                           zorder=1)  # vertical lines
            else:
                labels = list(group_categories["labels"])
                handles = legend_handles
                ax = plt.gca()
                ax.legend(
                    handles,
                    labels,
                    loc="upper right",
                    fontsize=8,
                    bbox_to_anchor=(1.5, 1),
                    frameon=False,
                    markerscale=1.5,
                )
            return fs
