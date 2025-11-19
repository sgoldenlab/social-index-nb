from io import BytesIO
import random
import matplotlib.pyplot as plt
import altair as alt
from typing import Union, Tuple
from scipy.stats import norm as normf
from sklearn import mixture
from scipy.stats import ks_2samp
from sklearn import decomposition
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import pandas as pd
import marimo as mo
import numpy as np
import string


from oss_app.utils import (
    make_categorical, mo_print, get_minmax, test_minmax, round_to_nearest
)


# helper functions
def set_global_font(font_name="Arial"):
    """Sets a global font for Altair charts.
    Example to apply the custom font configuration to the chart:
    alt.themes.register("my_custom_font_theme", set_global_font)
    alt.themes.enable("my_custom_font_theme")
    """
    return {
        "config": {
            "title": {'font': font_name},
            "axis": {
                "labelFont": font_name,
                "titleFont": font_name
            },
            "header": {
                "labelFont": font_name,
                "titleFont": font_name
            },
            "legend": {
                "labelFont": font_name,
                "titleFont": font_name
            },
            "view": {
                "strokeWidth": 0 
            }
        }
    }


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
    Uses the raw norm.pdf for the coefficient, then multiplies to 100 for percentage
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


def make_label_table(label_mapping):
    # mapping table layer for extra clarity
    mapping_table = pd.DataFrame(label_mapping.items(), columns=['metric', 'abbrev'])
    mapping_table['abbrev_x'] = [0]*len(mapping_table)
    mapping_table['full_x'] = [0.15]*len(mapping_table)
    mapping_table['y'] = -mapping_table.index
    mapscale = alt.Scale(domain=[mapping_table['y'].min()-0.5, 0.5])
    
    table_chart = alt.Chart(mapping_table).mark_text(
        align='right', 
        baseline='middle', 
        color='dimgrey',
        fontSize=14,
        fontWeight='bold'
    ).encode(
        x=alt.X('abbrev_x:Q', axis=None, scale=alt.Scale(domain=[-0.2, 2])), 
        y=alt.Y('y:Q', axis=None, scale=mapscale),
        text='abbrev:N',
    ) + alt.Chart(mapping_table).mark_text(
        align='left', 
        baseline='middle', 
        color='black',
        fontSize=12
    ).encode(
        x=alt.X('full_x:Q', axis=None, scale=alt.Scale(domain=[-0.2, 2])), 
        y=alt.Y('y:Q', axis=None, scale=mapscale),
        text='metric:N'
    )
    return table_chart


def create_diverging_colormap(base_colormap_name: str, n_colors: int = 256, reverse: bool=False, middle_white: bool=False):
    """
    Create a diverging colormap by mirroring the darker/low end of a base colormap.
    Creates: dark (negative, -1) -> medium (zero, 0) -> light (positive, +1).
    The darker half (0 to 0.5) is mirrored to create the positive side.
    
    Args:
        base_colormap_name: Name of the base matplotlib colormap
        n_colors: Number of colors in the diverging colormap
        reverse: Whether to reverse the base colormap before creating diverging version
        middle_white: If True, blend to white in the middle (for correlation matrices)
        
    Returns:
        List of hex color strings for the diverging colormap
    """
    base_cmap = cm.get_cmap(base_colormap_name)
    if reverse:
        base_cmap = base_cmap.reversed()
    
    if middle_white:
        # Create diverging colormap with white in the middle
        # Mirror one end of the colormap to create a symmetric diverging colormap
        n_half = n_colors // 2

        # Get colors from one half of the colormap
        # Which half depends on whether the colormap is reversed
        base_colors = base_cmap(np.linspace(0.15, 0.5, n_half))

        # Blend toward white in the middle
        # Create a smooth transition that mirrors around the center
        white = np.array([1.0, 1.0, 1.0, 1.0])  # White in RGBA

        # Blend base colors toward white (more white as we approach zero)
        blended_colors = []
        for i, color in enumerate(base_colors):
            # Blend factor: 0 (no white) at edges, 1 (all white) at center
            # Use a smoother curve for better blending
            # Lower exponent (e.g., 0.5-1.0) = more white, gentler transition
            # Higher exponent (e.g., 1.5-2.0) = less white, sharper transition
            blend_factor = (i / (n_half - 1)) ** 2.0 if n_half > 1 else 0
            blended = color[:3] * (1 - blend_factor) + white[:3] * blend_factor
            # Preserve alpha channel
            alpha = color[3] if len(color) > 3 else 1.0
            blended_colors.append(np.append(blended, alpha))

        # Create mirrored diverging colormap
        # The colormap will be symmetric: same color -> white -> same color
        diverging_colors = np.vstack([
            np.array(blended_colors),  # First half: color to white (-1 to 0)
            np.array(blended_colors[::-1])  # Second half: white to color (0 to 1), mirrored
        ])
    else:
        # Original method: mirror the darker half
        # Get colors from the darker half of the base colormap (0 to 0.5)
        # This represents the low/zero end
        n_half = n_colors // 2
        dark_half = base_cmap(np.linspace(0, 0.5, n_half))
        
        # Create diverging colormap by mirroring:
        # - First half: dark colors going from darkest to medium (negative correlations, -1 to 0)
        # - Second half: same colors reversed, going from medium to lightest (positive correlations, 0 to 1)
        # This creates: dark -> medium -> light
        diverging_colors = np.vstack([
            dark_half,  # Negative side: dark to medium (-1 to 0)
            np.flipud(dark_half)  # Positive side: medium to light (0 to 1), mirrored
        ])
    
    # Convert to hex
    hex_colors = [colors.rgb2hex(c[:3]) for c in diverging_colors]
    return hex_colors


def pca_correlation_heatmap(
    pca: PCA,
    princomps: np.ndarray,
    metric_labels: list[str],
    label_mapping: dict = None,
    n_comp: int = 3,
    colorset: ColorSet = None,
    hide_text: bool = False,
    use_letters: bool = True,
    table_orientation: str = "horizontal",  # "horizontal" or "vertical"
    colorbar_orientation: str = "vertical",  # "vertical" or "horizontal"
    colorbar_reverse: bool = False,
    colorbar_middle_white: bool = False,  # Use white in middle for correlation matrices
    set_size_params: tuple = None,  # (width, height) in inches
):
    """
    Create a correlation heatmap showing Pearson's correlation between 
    principal components and metrics.
    
    Args:
        pca: Fitted PCA object
        princomps: Principal component scores (n_samples, n_components)
        metric_labels: List of metric/variable names
        label_mapping: Dictionary mapping metric names to letter abbreviations
        n_comp: Number of principal components
        colorset: ColorSet object to derive colormap from
        hide_text: Whether to hide text labels
        use_letters: Whether to use letter abbreviations or full metric names
        table_orientation: "horizontal" (PCs as rows) or "vertical" (PCs as columns)
        colorbar_orientation: "vertical" or "horizontal"
        set_size_params: Tuple (width, height) in inches for plot size
        
    Returns:
        Altair chart object
    """
    # Calculate actual Pearson correlations between PCs and original metrics
    # We need the original scaled data - it should be passed or we can use loadings
    # For exact correlations, we calculate from data: corr(metric, PC) = loading * sqrt(explained_variance_ratio)
    # This is mathematically equivalent to calculating Pearson correlation directly
    
    # Get loadings
    loadings = pca.components_.T  # Shape: (n_features, n_components)
    
    # Create correlation matrix
    corr_data = []
    for i, metric in enumerate(metric_labels):
        for pc_idx in range(n_comp):
            # Calculate correlation: corr(metric, PC) = loading * sqrt(explained_variance_ratio)
            # This gives the Pearson correlation coefficient
            corr_value = loadings[i, pc_idx] * np.sqrt(pca.explained_variance_ratio_[pc_idx])
            
            # Determine label
            if use_letters and label_mapping:
                metric_label = label_mapping.get(metric, metric)
            else:
                metric_label = metric
            
            corr_data.append({
                'PC': f'PC{pc_idx+1}',
                'PC_num': pc_idx + 1,
                'Metric': metric_label,
                'Metric_full': metric,
                'Correlation': corr_value
            })
    
    corr_df = pd.DataFrame(corr_data)
    
    # Determine colormap
    if colorset:
        base_cmap_name = colorset.name
    else:
        base_cmap_name = "viridis"  # Default
    
    # Create diverging colormap
    diverging_colors = create_diverging_colormap(
        base_cmap_name, 
        n_colors=256, 
        reverse=colorbar_reverse,
        middle_white=colorbar_middle_white
    )
    
    # Determine orientation
    if table_orientation == "horizontal":
        x_field = 'Metric'
        y_field = 'PC'
        x_title = 'Metric'
        y_title = 'Principal Component'
    else:  # vertical
        x_field = 'PC'
        y_field = 'Metric'
        x_title = 'Principal Component'
        y_title = 'Metric'
    
    # Map user-friendly orientation to Altair legend positions
    if colorbar_orientation == "vertical":
        legend_orient = "right"  # Vertical colorbar on the right
        gradient_length = 200
        gradient_thickness = 20
    else:  # horizontal
        legend_orient = "top"  # Horizontal colorbar on top
        gradient_length = 300
        gradient_thickness = 15
    
    # Create heatmap
    base = alt.Chart(corr_df).mark_rect(
        stroke='black',
        strokeWidth=2
    ).encode(
        x=alt.X(f'{x_field}:N', title=x_title if not hide_text else '', 
                axis=alt.Axis(labelAngle=0 if table_orientation == "horizontal" else -45)),
        y=alt.Y(f'{y_field}:N', title=y_title if not hide_text else '',
                sort=alt.SortField('PC_num') if y_field == 'PC' else None),
        color=alt.Color('Correlation:Q',
            scale=alt.Scale(
                domain=[-1, 1],
                range=diverging_colors,
                nice=False
            ),
            legend=alt.Legend(
                title="Pearson's r" if not hide_text else '',
                orient=legend_orient,
                titleOrient=legend_orient,
                titleFontSize=12,
                titleColor='black',
                labelFontSize=12,
                labelColor='black',
                gradientLength=gradient_length,
                gradientThickness=gradient_thickness,
                tickCount=3,  # Show ticks at -1, 0, 1
                values=[-1, 0, 1],  # Explicitly set tick values
                format='.0d',
                columnPadding=20,
            )
        ),
        tooltip=[
            alt.Tooltip('Metric_full:N', title='Metric'),
            alt.Tooltip('PC:N', title='Principal Component'),
            alt.Tooltip('Correlation:Q', title="Pearson's r", format='.2f')
        ]
    )
    
    # Add text labels with correlation values
    text = alt.Chart(corr_df).mark_text(
        align='center',
        baseline='middle',
        color='black',
        fontSize=11,
        fontWeight='bold'
    ).encode(
        x=alt.X(f'{x_field}:N'),
        y=alt.Y(f'{y_field}:N',
                sort=alt.SortField('PC_num') if y_field == 'PC' else None),
        text=alt.Text('Correlation:Q', format='.2f')
    )
    
    # Combine heatmap and text
    heatmap = (base + text)
    
    # Set size
    if set_size_params:
        width, height = set_size_params
        # Convert inches to pixels (assuming ~100 DPI)
        width_px = int(width * 100)
        height_px = int(height * 100)
    else:
        # Default size based on orientation
        if table_orientation == "horizontal":
            width_px = 400
            height_px = 150
        else:
            width_px = 200
            height_px = 300
    
    heatmap = heatmap.properties(
        width=width_px,
        height=height_px,
        title=alt.Title('PCA Correlation Matrix', 
                       anchor='middle', baseline='top', 
                       fontSize=14, color='black') if not hide_text else None
    )
    
    # Apply styling
    styling = dict(
        configure=dict(background='white'),
        configure_view=dict(fill='white', stroke=None, strokeWidth=0, strokeOpacity=0)
    )
    
    heatmap = (
        heatmap
        .configure(**styling["configure"])
        .configure_view(**styling["configure_view"])
        .configure_axis(
            labelColor='black', titleColor='black', titleFontSize=12,
            tickColor='black', tickWidth=2,
            labelFontSize=12, labelFontWeight='bold', labelFont='arial',
            domainColor='black', domainWidth=2, grid=False
        )
    )
    
    if hide_text:
        heatmap = heatmap.configure_axis(labels=False, title=None)
        heatmap = heatmap.configure_title(text=None)
        heatmap = heatmap.configure_legend(title=None, labels=False)
    
    return heatmap


def si_scatter_plots(
    df_input: pd.DataFrame = pd.DataFrame(),
    metrics_included: list[str] | None = None,
    scaled=False,
    share_y=False,
    colorset: ColorSet | None = None,
    label_color: str = "white",
    hide_text=False,
    **scatter_kwargs
    ):

    assert not df_input.scaled_df.empty, "Input DataFrame `df_input` cannot be empty."
    if scaled:
        mo_print("Using scaled data for plotting.")
        df = df_input.scaled_df.copy()
    else:
        mo_print("Using raw data for plotting.")
        df = df_input.raw_df.copy()

    # are the metric variables explicitly defined?
    if not metrics_included:
        metrics_included = [
            m for m in df.columns if is_numeric_dtype(df[m]) and m != 'si_score']
    metric_labels = metrics_included

    # add subject id and group assignment
    subject_id_variable = df_input.subject_id_variable
    grouping_variable = df_input.grouping_variable

    # Color and Shape mapping
    group_categories = make_categorical(df[grouping_variable])
    if colorset is None:
        colorset = ColorSet(
            color_name="viridis_r",
            metric_name='si_score',
            grouping_variable=grouping_variable,
            group_name=group_categories["labels"][0],
            data=df
        )
    fc_cols = [row for row in colorset.hex_colors]
    style_mapping = {
        label: shape for label, shape in zip(group_categories['labels'].tolist(), ['triangle-down', 'circle'])
    }
    size_mapping = {
        label: size for label, size in zip(group_categories['labels'], [180,140])
    }

    # plotting
    random.seed(123)  # for consistent output
    x_index = group_categories['codes']
    plot_settings =  scatter_kwargs.get('plot_settings', {})
    if scaled:  # auto-adjust axes if scaled data, which should be in shared y-ranges
        plot_settings = {}

    if plot_settings == {}:

        current_minmax = (0, 0)
        if share_y:
            for v in metrics_included:
                current_minmax = test_minmax(df[v].values, current_minmax, debug=False)
            for v in metrics_included:
                plot_settings.update({
                    v: dict(values=df[v].values, title=v, ylims=current_minmax),
                    'shared_ylims': current_minmax,
                })
        else:
            for v in metrics_included:
                plot_settings.update({
                    v: dict(values=df[v].values, title=v, ylims=(min(df[v].values), max(df[v].values)))
                })
    df['color'] = fc_cols
    df['shape'] = df[grouping_variable].map(style_mapping)
    df['size'] = df[grouping_variable].map(size_mapping)

    # Prepare data for Altair
    df_long = df.copy()
    df_long[subject_id_variable] = df[subject_id_variable]
    df_long[grouping_variable] = df[grouping_variable]
    df_long['color'] = fc_cols
    df_long['shape'] = df_long[grouping_variable].map(style_mapping)
    df_long['size'] = df_long[grouping_variable].map(size_mapping)


    # Melt the DataFrame to long format for faceting
    df_melted = df_long.melt(
        id_vars=[subject_id_variable, grouping_variable, 'color', 'shape', 'size'],
        value_vars=metrics_included,
        var_name='metric',
        value_name='value'
    )
    df_melted['title'] = df_melted['metric'].map(lambda m: plot_settings[m]['title'])
    df_melted['high'] = df_melted.apply(lambda row: plot_settings[row['metric']]['ylims'][1], axis=1)

    # Base chart for a single facet
    chart = alt.Chart(df).mark_point(
        filled=True,
        opacity=1,
        stroke='black',
        strokeWidth=0.5
    ).encode(
        x=alt.X(
            f'{grouping_variable}:N',
            axis=alt.Axis(
                title=None,
                labels=False,
                ticks=False,
                domain=False,
                grid=False,
                labelAngle=0
            )),
        y=alt.Y(
            # 'value:Q',
            alt.repeat('column'), type='quantitative',
            scale=alt.Scale(
                nice=1 if not scaled else 2,  # if scaled data, don't adjust to "nice" numbers
            ),
            axis=alt.Axis(
                # title='raw values',  # use the mapped title),
                titleFontSize=10 if not hide_text else 0,
                labels=not hide_text,
                labelColor=label_color,
                grid=False,
                tickCount=1 if not scaled else 2,
                offset=10,
                # values=alt.range(alt.value(0), alt.datum.high)
            )),
        color=alt.Color(
            'color:N', scale=None,),
        shape=alt.Shape('shape:N',
                        scale=alt.Scale(
                            domain=list(style_mapping.values()),
                            range=list(style_mapping.values())
                            )),
        size=alt.Size('size:Q', scale=None),   # Use pre-mapped sizes
        # tooltip=[subject_id_variable, grouping_variable, 'metric', 'value']
    ).transform_calculate(
        # Adjust the scale to control the spread of the jitter.
        x_jittered='datum.x + 0.25+(0.5*datum.x)'  # standard jitter
    ).repeat(column=metrics_included)

    # Apply final configurations
    final_chart = chart.configure_view(  # facet_chart.configure_view(
        stroke=None  # Remove border around each plot
    ).configure_axis(
        labelFont='arial',
        titleFont='arial',
    ).configure_axisLeft(labelColor=label_color)

    return final_chart


def plot_colorbar(
    df_input: pd.DataFrame = pd.DataFrame(),
    colorset: ColorSet | None = None,
    ):

    assert not df_input.scaled_df.empty, "Input DataFrame `df_input` cannot be empty."
    df = df_input.scaled_df.copy()
    # mo_print(df)

    # Color and Shape mapping
    grouping_variable = df_input.grouping_variable
    group_categories = make_categorical(df[grouping_variable])
    if colorset is None:
        colorset = ColorSet(
            color_name="viridis_r",
            metric_name='si_score',
            grouping_variable=grouping_variable,
            group_name=group_categories["labels"][0],
            data=df
        )
    fc_cols = [row for row in colorset.hex_colors]
    df['color'] = fc_cols

    colorbar = alt.Chart(df).mark_rect(opacity=0).encode(
        alt.Y('si_score:Q', axis=None),
        color=alt.Color('si_score:Q', scale=alt.Scale(range=sorted(fc_cols,reverse=True)),
        legend=alt.Legend(
            title='Index Score', tickCount=3, orient='left',
            gradientStrokeColor='black', gradientStrokeWidth=1,
            labelOffset=5, type='gradient'
        ))
    ).interactive(False).properties(width=10)
    return colorbar


def pca_biplot_altair(
    df_input=pd.DataFrame(),
    metrics_included: list[str] | None = None,
    labels="",
    pca_inputs: list[Union[PCA, np.ndarray, ...]] = None,
    n_comp=3,
    pcs: list[int] | None = None,
    colorset: ColorSet | None = None,
    hide_text=False,
    loading_scale: str = "data_range",  # "variance", "data_range", or "correlation"
    show_heatmap: bool = True,  # Whether to show correlation heatmap
    heatmap_use_letters: bool = True,  # Use letter abbreviations in heatmap
    heatmap_orientation: str = "horizontal",  # "horizontal" or "vertical"
    heatmap_colorbar_orientation: str = "vertical",  # "vertical" or "horizontal"
    heatmap_colorbar_reverse: bool = False,
    heatmap_colorbar_middle_white: bool = False,  # Use white in middle for correlation matrices
    heatmap_size: tuple = None,  # (width, height) in inches
    **scatter_kwargs,
):
    """
    Generates a PCA biplot using Altair.
    """
    
    assert not df_input.scaled_df.empty, "Input DataFrame `df_input` cannot be empty."
    df = df_input.scaled_df.copy()

    # are the metric variables explicitly defined?
    if not metrics_included:
        metrics_included = [
            m for m in df.columns if is_numeric_dtype(df[m]) and m != 'si_score']
    metric_labels = metrics_included

    # PCA
    if not pca_inputs:
        pca, princomps, pc_evr = do_pca(df[metric_labels], n_comp)
    else:
        pca, princomps, pc_evr = pca_inputs

    if pcs is None:
        pcs = [0, 1]
    pc_x, pc_y = pcs[0], pcs[1]

    # Create a DataFrame with principal components
    pc_df = pd.DataFrame(
        princomps[:, [pc_x, pc_y]],
        columns=[f'PC{pc_x+1}', f'PC{pc_y+1}'],
        index=df.index
    )
    # add subject id and group assignment
    subject_id_variable = df_input.subject_id_variable
    grouping_variable = df_input.grouping_variable
    pc_df[subject_id_variable] = df[subject_id_variable]
    pc_df[grouping_variable] = df[grouping_variable]

    # Color and Shape mapping
    group_categories = make_categorical(df[grouping_variable])
    if colorset is None:
        colorset = ColorSet(
            color_name="viridis_r",
            metric_name='si_score',
            grouping_variable=grouping_variable,
            group_name=group_categories["labels"][0],
            data=df
        )
    # color for each datapoint, based on 'si_score' value
    pc_df['color'] = [row for row in colorset.hex_colors]
    # marker style/shape based on group
    style_mapping = {
        label: shape for label, shape in zip(group_categories['labels'].tolist(), ['triangle-down', 'circle'])
    }
    pc_df['shape'] = df[grouping_variable].map(style_mapping)
    # marker size based on group
    size_mapping = {
        label: size for label, size in zip(group_categories['labels'], [180,140])
    }
    pc_df['size'] = df[grouping_variable].map(size_mapping)
    
    max_abs_value = max(np.abs(pc_df.iloc[:,pc_x]).max(), np.abs(pc_df.iloc[:,pc_y]).max())
    shared_scale = alt.Scale(domain=[-max_abs_value-0.2, max_abs_value+0.2])
    
    # Scatter plot of principal components
    zero_lines = alt.Chart(  # Add dashed line for x=0
        pd.DataFrame({'x': [0]})
        ).mark_rule(strokeDash=[5, 5], strokeWidth=2,
        ).encode(
            x='x:Q',
            color=alt.value('gray') # Optional: set a color for the line
        ) + alt.Chart(  # Add dashed line for y=0
            pd.DataFrame({'y': [0]})
        ).mark_rule(strokeDash=[5, 5], strokeWidth=2
        ).encode(
            y='y:Q',
            color=alt.value('gray') # Optional: set a color for the line
    )

    hover = alt.selection_point(on='pointerover', nearest=False, empty=False, resolve='intersect')
    when_hover = alt.when(hover)

    scatter = alt.Chart(pc_df, width=200, height=200).mark_point(
        filled=True, opacity=1
    ).encode(
        x=alt.X(f'PC{pc_x+1}:Q', title=f'PC{pc_x+1}  ({pc_evr[pc_x]:.2%})', 
                scale=shared_scale, axis=alt.Axis(orient='bottom', labels=False, ticks=True)),
        y=alt.Y(f'PC{pc_y+1}:Q', title=f'PC{pc_y+1}  ({pc_evr[pc_y]:.2%})', 
                scale=shared_scale, axis=alt.Axis(orient='left', labels=False if hide_text else True, ticks=True)),
        color=alt.Color('color:N', legend=None).scale(
            domain=colorset.hex_colors, range=colorset.hex_colors),
        stroke=when_hover.then(alt.value('black')).otherwise(alt.value('transparent')), # highlight point
        order=when_hover.then(alt.value(1)).otherwise(alt.value(0)),  # bring hovered point to front
        shape=alt.Shape('shape:N',
                        legend=alt.Legend(
                            orient='right', offset=30,
                            title=grouping_variable, symbolType='symbol', 
                            labelColor='black', labelFontSize=12)
                    ).scale(domain=group_categories["labels"].tolist(), range=['triangle-down', 'circle']),
        size=alt.Size('size:Q', scale=None),
        tooltip=[subject_id_variable, grouping_variable]
    ).add_params(hover)

    # Loadings plot - with different scaling options
    if loading_scale == "correlation":
        # Scale by sqrt of explained variance ratio for correlation interpretation
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_ratio_)
    elif loading_scale == "data_range":
        # Scale loadings to be proportional to PC score range (may have better visibility)
        # First calculate variance-scaled loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Get the range of PC scores for the displayed components
        pc_scores_displayed = princomps[:, [pc_x, pc_y]]
        pc_range = np.abs(pc_scores_displayed).max()
        
        # Get the range of loadings for the displayed components
        loadings_displayed = loadings[:, [pc_x, pc_y]]
        loading_range = np.abs(loadings_displayed).max()
        
        # Scale loadings to be 80% of PC range (adjustable)
        if loading_range > 0:
            scale_factor = (pc_range * 0.8) / loading_range
            loadings = loadings * scale_factor
    else:  # "variance" - standard variance scaling
        # Variance scaling: loadings represent variance contributions
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(n_comp)],
        index=metric_labels
    )
    
    # Arrows for loadings
    arrows_data = []
    for i, metric in enumerate(metric_labels):
        arrows_data.append({
            'metric': metric,
            'origin': 0,
            'x': loadings[i, pc_x],
            'y': loadings[i, pc_y],
            'angle': 90.0-np.round(np.arctan2(loadings[i, pc_y], loadings[i, pc_x], dtype=float) * 180.0 / np.pi, 5)
        })
    arrows_df = pd.DataFrame(arrows_data)

    # mapping labels to loadings
    label_mapping = {
        original: letter for original, letter in zip(arrows_df['metric'], string.ascii_lowercase)
    }
    arrows_df['abbrev'] = arrows_df['metric'].map(label_mapping)
    
    arrows = alt.Chart(arrows_df).mark_rule(color='indianred', opacity=1, strokeWidth=3).encode(
        x=alt.X('origin:Q', scale=shared_scale),
        y=alt.Y('origin:Q', scale=shared_scale),
        x2='x:Q',
        y2='y:Q'
    ) + alt.Chart(arrows_df).mark_point(shape='triangle', size=150, opacity=1,
        color='indianred', filled=True,
    ).encode(
        x=alt.X('x:Q', scale=shared_scale),
        y=alt.Y('y:Q', scale=shared_scale),
        angle=alt.Angle('angle:Q', scale=alt.Scale(domain=[-180,180], range=[-180,180])),
        tooltip=[alt.Tooltip('metric', title='Metric name =')],
    )

    arrow_labels = alt.Chart(arrows_df).mark_text(
        align='left',
        dx=-12,
        dy=-8,
        color='black',
        size=24,
        clip=False  # Add this to prevent clipping
    ).encode(
        x=alt.X('x_jittered:Q', scale=shared_scale),
        y=alt.Y('y_jittered:Q', scale=shared_scale),
        text='abbrev:N',
    ).transform_calculate(
        # Adjust the scale to control the spread of the jitter.
        x_jittered='datum.x + (datum.x > 0 ? 0.25+abs(0.5*datum.x) : -0.25-abs(0.5*datum.x))',
        y_jittered='datum.y + (datum.y > 0.25 ? 0.25*datum.y : -0.25*datum.y)'
    )

    chart = (zero_lines + scatter + arrows + arrow_labels).properties(
        title=alt.Title(f'PCA biplot  [ {grouping_variable=} ]', 
                        anchor='middle', baseline='top', fontSize=14, color='black'), 
        width=200, height=200
    ).interactive()

    mapping_chart = make_label_table(label_mapping).properties(
        title=alt.Title('Labels', anchor='start', baseline='top', fontSize=14, color='black'),
        height=25*len(label_mapping)
    )

    # chart = alt.hconcat(chart, mapping_chart).properties(
    #     width=250,  # Set width to 600 pixels
    #     height=500  # Set height to 400 pixels
    # )
    # general styling
    styling = dict(
        configure=dict(
            background='white'),
        configure_view=dict(
            fill='white', stroke=None, strokeWidth=0, strokeOpacity=0)
    )

    chart = (
        chart
        .configure(**styling["configure"])
        .configure_view(**styling["configure_view"])
        .configure_axis(
                labelColor='black', titleColor='black', titleFontSize=14,
                tickColor='black', tickWidth=2, 
                labelFontSize=12, labelFontWeight='bold', labelFont='arial',
                labelFlush=False, labelPadding=5,
                domainColor='black', domainWidth=2, grid=False  #gridColor='black'#gridDash=[2,2]
        )).configure_axisTop(
            domain=False
        ).configure_axisRight(
            domain=False
        )

    mapping_chart = (
        mapping_chart
        .configure(**styling["configure"])
        .configure_view(**styling["configure_view"])
    )

    if hide_text:
        chart = chart.configure_axis(labels=False, title=None)
        chart = chart.configure_title(text=None)
        chart = chart.configure_legend(title=None, labels=False)

    # Create correlation heatmap if requested
    heatmap_chart = None
    if show_heatmap:
        heatmap_chart = pca_correlation_heatmap(
            pca=pca,
            princomps=princomps,
            metric_labels=metric_labels,
            label_mapping=label_mapping,
            n_comp=n_comp,
            colorset=colorset,
            hide_text=hide_text,
            use_letters=heatmap_use_letters,
            table_orientation=heatmap_orientation,
            colorbar_orientation=heatmap_colorbar_orientation,
            colorbar_reverse=heatmap_colorbar_reverse,
            colorbar_middle_white=heatmap_colorbar_middle_white,
            set_size_params=heatmap_size
        )

    return chart, mapping_chart, heatmap_chart