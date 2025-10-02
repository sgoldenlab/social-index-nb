# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():

    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # import pandera.pandas as pa
    from pathlib import Path
    import altair as alt
    import asyncio
    import json
    from os import mkdir
    from datetime import datetime

    return Path, alt, asyncio, datetime, json, mo, np, pd, plt


@app.cell
def _():
    from oss_app import oss_func_simple as sf
    from oss_app import utils as ut
    from oss_app import dataset as ds
    from oss_app.utils import make_categorical
    from oss_app.plotting import show_color, show_colormap, colormap_to_hex
    return ds, make_categorical, show_color, show_colormap


@app.cell
def _(mo):
    mo.md(r"""# Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""## file selection""")
    return


@app.cell
def _(Path, mo):
    # set starting path and dynamic state to retrieve, for returning to default
    # change if need to look for file elsewhere
    starting_path = Path(".").resolve()
    reset_button = mo.ui.run_button(label='Return to initial path')

    checkbox_style = "<span style='display:inline;vertical-align:baseline;color:coral; font-size:14px'>"
    #     'justify-self':'left','padding':'1px 1px 1px 5px'>

    overwrite_cbox = mo.ui.checkbox(
        label=checkbox_style+"Overwrite outputs</span>", value=False)
    return checkbox_style, overwrite_cbox, reset_button, starting_path


@app.cell
def _(mo, starting_path):
    mo.stop(not starting_path.exists(),
            "Starting path for file browser is not valid.")
    # file selection
    browser = mo.ui.file_browser(
        initial_path=starting_path,
        filetypes=['.csv'],
        selection_mode='file',
        multiple=False,
        restrict_navigation=False,  # prevent from moving out of input csv folder
        label='''###Select CSV file to import.<br /></h3><font color="khaki">If file not visible below, make sure it is inside the "raw_csv" folder.''',
    )

    # file drag and drop
    file_drop_ui = mo.ui.file(
        filetypes=['.csv'],
        multiple=False,
        kind='area',
        label='###Drop or select a CSV file to open.<br /></h3>Then click "Confirm Selection" to import, if it has issues importing file, use the file explorer.',
    )
    file_drop = file_drop_ui.form(
        bordered=True, loading=False,
        submit_button_label="Confirm Selection"
    )

    # file selector
    file = mo.md('''
        {browser}
        ''').batch(browser=browser,
                   ).form(
        bordered=True, loading=False,
        submit_button_label="Confirm Selection"
    )
    # <span style='display:inline; float:right; box-sizing:border-box; border:1px solid coral; padding-left:5px'>{overwrite}<br>{append}</span>
    # selections for overwriting or appending outputs
    return file, file_drop


@app.cell
def _(checkbox_style, mo, overwrite_cbox):
    append_cbox = mo.ui.checkbox(label=checkbox_style+"Append outputs</span>",
                                 value=False, disabled=True if overwrite_cbox.value else False)
    return (append_cbox,)


@app.cell
async def _(
    asyncio,
    file,
    file_drop,
    file_ui,
    mo,
    reset_button,
    starting_path,
):
    mo.stop(not starting_path.exists())
    # reset gui
    mo.output.append(mo.ui.tabs({
        'file explorer': file,
        'drag & drop': file_drop.style({'display': 'flex-grow', 'width': '400px', 'height': '350px', 'justify-self': 'center'})
    }))

    if reset_button.value:
        mo.output.clear()
        with mo.status.spinner(title="Reloading GUI...") as _spinner:
            # task=asyncio.create_task(mo.output.clear())

            # mo.output.append(_spinner)
            _spinner.update(subtitle="Clearing...")
            await asyncio.sleep(1)

            _spinner.update(subtitle="Reloading...")
            await asyncio.sleep(1.5)
            # await asyncio.run(mo.output.clear())
            # await task
            _spinner.update(title="Done...")
            mo.output.replace(file_ui)

    return


@app.cell
def _(append_cbox, mo, overwrite_cbox):
    overwrite_choice = mo.md(f"""
            <span style='display:inline; float:left; width:150px; box-sizing:border-box; border:1px solid coral; padding-left:5px'>{overwrite_cbox}</span>
            """)

    append_choice = mo.md(f"""
            <span style='display:inline; float:left; width:150px; box-sizing:border-box; border:1px solid coral; padding-left:5px'>{append_cbox}</span>
            """)
    return append_choice, overwrite_choice


@app.cell
def _(append_choice, mo, overwrite_choice):

    mo.output.append(
        mo.vstack([
            mo.md("<br><h3> Overwrite or append selections</h3>"),
            mo.hstack([overwrite_choice, mo.md("Replace output folders and files if they exist.")],
                      justify='start', align='center', widths=[0.15, 0.85]),
            mo.hstack([append_choice, mo.md("Disabled if overwriting. Appends numbers to output folders and files if they exist.")],
                      justify='start', align='center', widths=[0.15, 0.85]),
        ])
    )
    return


@app.cell
def _(Path, file, file_drop, mo, overwrite_cbox):
    # Prevent progression if no file selected, mo.stop prevents code below from running.
    file_not_chosen = (file.value is None) and (file_drop.value is None)
    mo.stop(file_not_chosen, mo.md(
        "###**Confirm file selection to continue...**"))

    match not file_not_chosen:
        case _x if _x and not file.value:  # file drop upload used
            file_choice = file_drop.value[0]
            filename = file_choice.name
            filepath = Path(filename).resolve()
            with mo.redirect_stdout():
                assert filepath.exists(
                ), f'Issues resolving path of uploaded file : {filepath} does not exist.\nPlease use file explorer.'
            filepath_parent = Path(filepath).parent
        case _y if _y and not file_drop.value:  # file explorer upload used
            file_choice = file.value["browser"][0]
            filepath = Path(file_choice.id)
            filename = Path(filepath).name

    filepath_parent = Path(filepath).parent
    overwrite = overwrite_cbox.value
    if overwrite:
        _notice = mo.md(f"""
        /// warning | Overwrite set to True, outputs will be replaced.
        """)
    else:
        _notice = mo.md(f"""
        /// tip | Overwrite set to False, new outputs will be created, with appended numbers if already present.
        """)
    mo.output.append(_notice)
    return file_not_chosen, filename, filepath, filepath_parent, overwrite


@app.cell
def _(Path, datetime, file_not_chosen, filename, filepath_parent, mo):
    mo.stop(file_not_chosen)
    create_btn = mo.ui.run_button(label='Create new output folders')

    # output folder will be in input file folder
    # today_dt = datetime.today().strftime('%y%m%d_%Hh%Mm')  # date and time stamp
    today_dt = datetime.today().strftime('%y%m%d')  # date stamp only
    save_path = filepath_parent / f'{Path(filename).stem}_{today_dt}'

    # check for previously saved params file
    save_folder_found = save_path.exists()
    params_found = (params_file := list(
        filepath_parent.rglob('params*.json'))) != []
    load_btn = mo.ui.run_button(
        label='Load params file', disabled=not params_found)  # load params button
    loaded_params, set_loaded_params = mo.state(False)

    return (
        create_btn,
        load_btn,
        loaded_params,
        params_file,
        params_found,
        save_folder_found,
        save_path,
        set_loaded_params,
        today_dt,
    )


@app.cell
def _(mo, overwrite, params_file, params_found, save_folder_found, save_path):
    with mo.redirect_stdout():
        if not overwrite:
            if save_folder_found and not params_found:
                print(
                    f'Found save folder at:<br> {save_path}, <br>but no `params` json file found.')
            elif params_found:
                # params_file = params_file[0]
                param_file_dict = {f.name: f for f in params_file}
                param_files = list(param_file_dict.keys())
                select_params = mo.ui.dropdown(
                    param_files, value=param_files[0] if params_found else None)

                print(f'Found params file(s).')
                print(f'Load previous parameters or create new output folder?')

            else:
                print(f'No previous params file found.')
                print(f'Click button to create new output folder.')
    return param_file_dict, select_params


@app.cell
def _(create_btn, load_btn, mo, select_params):
    # if not overwriting and previous params found; load or create new?
    mo.output.append(mo.vstack([
        create_btn,
        mo.hstack([load_btn, select_params.left()])
    ]))
    return


@app.cell
def _(mo):
    def display_selections_markdown(params_output: dict, title: str = "Selections"):
        if not params_output:
            return mo.md("---\n\n## No selections have been confirmed yet.")

        md_content = f"---\n\n## <span style='color:lightcoral'>{title}\n\n"

        # Helper to format list or single item
        def format_value(key, value):
            if value is None or (isinstance(value, list) and not value):
                return f"* {key.replace('_', ' ').title()}: _Not selected_"
            elif isinstance(value, list):
                return f"* {key.replace('_', ' ').title()}: &nbsp; <span style='color:coral; font-family:monospace'>{', '.join(value)}</span>"
            else:
                return f"* {key.replace('_', ' ').title()}: &nbsp;  `{value}`"

        filename = params_output["filename"]
        fformat = "<span style='color:coral; font-family: monospace'>"
        md_content += "#### Input file:\n&nbsp; " + f'"`{filename}`"'
        md_content += "\n#### Variables:"
        md_content += "\n" + \
            format_value("subject_id_variable",
                         params_output.get("subject_id_variable"))
        md_content += "\n" + \
            format_value("sex_variable",
                         params_output.get("sex_variable"))
        md_content += "\n" + \
            format_value("grouping_variable",
                         params_output.get("grouping_variable"))
        md_content += "\n" + \
            format_value("index_variables",
                         params_output.get("index_variables"))
        md_content += "\n" + \
            format_value("metric_variables",
                         params_output.get("metric_variables"))

        md_content += "\n#### Filters:"
        filters = params_output.get("filters", {})
        if filters == {}:
            md_content += "\n* " + f'{fformat}No filters added/loaded</span>'
        else:
            for fkey, fval in filters.items():
                md_content += "\n* " + \
                    f'{fformat}{fkey}</span>" &nbsp;== &nbsp; "{fformat}{fval}</span>"'

        md_content += "\n#### Colors:"
        colors = params_output.get("colors", {})
        if colors == {}:
            md_content += "\n* " + f'{fformat}No colors chosen</span>'
        else:
            for group_col in colors:
                for fkey, fval in group_col.items():
                    md_content += "\n* " + \
                        f'"{fformat}{fkey}</span>" &nbsp;== &nbsp; "{fformat}{fval}</span>"'
        return mo.md(md_content)
    return (display_selections_markdown,)


@app.cell
def _(
    display_selections_markdown,
    json,
    load_btn,
    loaded_params,
    mo,
    param_file_dict,
    select_params,
    set_loaded_params,
):
    if load_btn.value:  # load params pressed
        param_file = param_file_dict[select_params.value]
        with open(param_file, 'r') as _file:
            previous_params = json.load(_file)
        set_loaded_params(True)

    if not loaded_params():
        previous_params = None

    mo.stop(not load_btn.value)
    mo.ui.tabs({
        'Parameters loaded': display_selections_markdown(previous_params, 'Loaded Parameters'),
        'JSON file': mo.json(previous_params, label='Loaded PARAMS.JSON')
    })
    return (previous_params,)


@app.cell
def _(
    Path,
    create_btn,
    filename,
    filepath_parent,
    loaded_params,
    mo,
    overwrite,
    save_path,
    today_dt,
):
    mo.stop(create_btn.value == False or loaded_params())
    _save_path = save_path  # make local copy of variable
    match _save_path.exists():
        case x if x and overwrite:  # overwrite == True
            with mo.redirect_stdout():
                print(
                    f'Output folder exists already in:<br> {_save_path},<br>output folder will be replaced.')
        case y if y and (not overwrite):  # overwrite == False
            with mo.redirect_stdout():
                print(
                    f"""Output folder exists already in:<br> {_save_path}.  
                    A new appended output folder will be made.""")
            _i = 0
            while _save_path.exists():  # if file already exist, append numbers
                _save_path = filepath_parent / \
                    f'{Path(filename).stem}_{today_dt}_{_i}'
                _i += 1
    # create output folder
    # parents False as csv file should be present in existing path
    _save_path.mkdir(exist_ok=True, parents=False)
    with mo.redirect_stdout():
        print(f'Created output folder at:<br> {_save_path}')

    return


@app.cell
def _(create_btn, filepath, loaded_params, mo, pd):
    mo.stop(create_btn.value == False and not loaded_params())
    df = pd.read_csv(filepath)
    choice = mo.ui.switch(False, label="### Advanced")

    mo.vstack([
        mo.md('<br>'),
        mo.md("### CSV file loaded. Use the GUI below to explore the dataset."),
        mo.accordion(
            {'Click here to reveal data explorer': mo.ui.data_explorer(df)}),
    ])
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    <br>

    ---
    ## <span style='color:lightcoral'> Variables and metrics selection
    ...
    """
    )
    return


@app.cell
def _(df, loaded_params, mo, previous_params):
    if not loaded_params():
        subject_id_initial = []
        sex_variable_initial = []
        grouping_variable_initial = []
        index_variables_initial = []
        metrics_variables_initial = []
    else:
        subject_id_initial = [previous_params["subject_id_variable"]]
        sex_variable_initial = [previous_params["sex_variable"]]
        grouping_variable_initial = [previous_params["grouping_variable"]]
        index_variables_initial = previous_params["index_variables"]
        metrics_variables_initial = previous_params["metric_variables"]

    # Create dropdowns for selecting columns, or fill in with loaded parameters
    subject_id_selector = mo.ui.multiselect(
        options=df.columns.tolist(), label="ID Variable", max_selections=1,
        value=subject_id_initial
    )

    sex_variable_selector = mo.ui.multiselect(
        options=df.columns.tolist(), label="Sex Variable", max_selections=1,
        value=sex_variable_initial
    )

    grouping_variable_selector = mo.ui.multiselect(
        options=df.columns.tolist(), label="Grouping Variable", max_selections=1,
        value=grouping_variable_initial
    )
    return (
        grouping_variable_selector,
        index_variables_initial,
        metrics_variables_initial,
        sex_variable_selector,
        subject_id_selector,
    )


@app.cell
def _(mo):
    def display_selections(selection: list = [], multiple=False):
        if selection == []:
            return "&nbsp;"
        elif len(selection) < 2:
            return selection[0]
        elif multiple:  # multiple selection dropdowns
            return ',<br>'.join([i for i in selection])
        else:
            return ',<br>'.join([i[0] for i in selection])

    def description(description_text: str):
        md_content = mo.md(
            # Use <small> tag for smaller text
            f"<medium>{description_text}</medium>"
        ).style({"color": '#AAAAAA'})

        return md_content
    return description, display_selections


@app.cell
def _(mo):
    mo.hstack([
        mo.md(f"### Select Main Variables for Analysis").style(
            {"flex-grow": "1"}),
        mo.md("### Descriptions").style({"flex-grow": "1"})
    ])
    return


@app.cell
def _(description, display_selections, mo, subject_id_selector):
    mo.output.append(
        mo.hstack([
            mo.vstack([
                subject_id_selector,
                mo.md(display_selections(subject_id_selector.value)).style(
                    {"color": "lightlavender"})
            ]),
            description(
                "Select the column that uniquely identifies each subject in the dataset.")
        ],
            # 30% for the left column, 70% for the right
            widths=[0.3, 0.7],
            # Align the content of the hstack to the end (right)
            justify="end",
            align="start",     # Align content of this hstack to the top
            wrap=True),
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _(description, display_selections, mo, sex_variable_selector):
    mo.output.append(
        mo.hstack([
            mo.vstack([
                sex_variable_selector,
                mo.md(display_selections(sex_variable_selector.value)).style(
                    {"color": "lightlavender"})
            ]),
            description(
                "Select the column that defines the sex of each subject in the dataset.")
        ],
            # 30% for the left column, 70% for the right
            widths=[0.3, 0.7],
            # Align the content of the hstack to the end (right)
            justify="end",
            align="start",     # Align content of this hstack to the top
            wrap=True),
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _(description, display_selections, grouping_variable_selector, mo):
    mo.output.append(
        mo.hstack([
            mo.vstack([
                grouping_variable_selector,
                mo.md(display_selections(grouping_variable_selector.value)).style(
                    {"color": "lightlavender"})
            ]),
            description("""Select the column that defines the grouping or category for comparison within your data (e.g., condition [control vs treatment], delay[short vs long]).  
             <span style='font-size:0.8em; color:#888888'>Only one can be chosen to compare at a time for now, use filters below to restrict comparisons if many grouping variables are present.  \nFor example, if grouping by condition but don't want to collapse across sex, filter for male or female.</span>""")
        ],
            # 30% for the left column, 70% for the right
            widths=[0.3, 0.7],
            # Align the content of the hstack to the end (right)
            justify="end",
            align="start",     # Align content of this hstack to the top
            wrap=True),
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _(grouping_variable_selector, sex_variable_selector, subject_id_selector):
    selected_idx_options = [x.value for x in [
        subject_id_selector, sex_variable_selector, grouping_variable_selector]]
    return (selected_idx_options,)


@app.cell
def _(df, mo, selected_idx_options):
    mo.stop(any(selection==[] for selection in selected_idx_options), mo.md("###Select index variables to continue..."))

    filtered_idx_options = [col for col in df.columns.to_list(
    ) if col not in set([opt[0] for opt in selected_idx_options])]
    return (filtered_idx_options,)


@app.cell
def _(filtered_idx_options, index_variables_initial, mo):
    index_column_selector = mo.ui.multiselect(
        options=filtered_idx_options, label="Other Indices",
        value=index_variables_initial
    )
    return (index_column_selector,)


@app.cell
def _(df, index_column_selector, selected_idx_options):
    indices = index_column_selector.value
    selected_options = set(
        [i[0] if isinstance(i, list) else i for i in selected_idx_options+indices])

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    filtered_options = [option for option in df.columns.tolist() if (
        option not in selected_options) or (option in numeric_columns)]
    return (filtered_options,)


@app.cell
def _(filtered_options, metrics_variables_initial, mo):
    # metric selector (index variable choices removed)
    metric_columns_selector = mo.ui.multiselect(
        options=filtered_options, label="Metrics",
        value=metrics_variables_initial
    )
    return (metric_columns_selector,)


@app.cell
def _(loaded_params, mo):
    # button to confirm choices
    submit_choices = mo.ui.run_button(
        label="Confirm Selections", kind='warn', disabled=loaded_params())
    var_choices = {}
    return submit_choices, var_choices


@app.cell
def _(mo):
    mo.hstack([
        mo.md(f"### Select Metrics and Additional Index Variables for Analysis").style(
            {"flex-grow": "1"}),
        mo.md("### Descriptions").style({"flex-grow": "1"})
    ])
    return


@app.cell
def _(description, display_selections, index_column_selector, mo):
    mo.output.append(
        mo.hstack([
            mo.vstack([
                index_column_selector,
                mo.md(display_selections(index_column_selector.value,
                      multiple=True)).style({"color": "lightlavender"})
            ]),
            description("""Select additional columns that serve as unique (sub)groups in the data.  
                e.g., secondary condition, age, delay, etc.""")
        ],
            # 30% for the left column, 70% for the right
            widths=[0.3, 0.7],
            # Align the content of the hstack to the end (right)
            justify="end",
            align="start",     # Align content of this hstack to the top
            wrap=True),
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _(description, display_selections, metric_columns_selector, mo):
    mo.output.append(
        mo.hstack([
            mo.vstack([
                metric_columns_selector,
                mo.md(f"<span style='display: block; font-size:0.9em; line-height:1.5em'>" + display_selections(
                    metric_columns_selector.value, multiple=True)).style({"color": "lightlavender"})
            ]),
            description("""Select the columns containing numerical data to analyze as behavioral metrics.  
                These will be used for plotting and statistical comparisons.""")
        ],
            # 30% for the left column, 70% for the right
            widths=[0.3, 0.7],
            # Align the content of the hstack to the end (right)
            justify="end",
            align="start",     # Align content of this hstack to the top
            wrap=True),
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _(submit_choices):
    submit_choices.style(
        {'width': 'fit-content', 'padding': '20px 20px'}).right()
    return


@app.cell
def _(
    grouping_variable_selector,
    index_column_selector,
    loaded_params,
    metric_columns_selector,
    mo,
    sex_variable_selector,
    subject_id_selector,
    submit_choices,
    var_choices,
):
    mo.stop((submit_choices.value is False) and (not loaded_params()))
    var_choices.update({
        "subject_id_variable": subject_id_selector.value[0],
        "grouping_variable": grouping_variable_selector.value[0],
        "sex_variable": sex_variable_selector.value[0],
        "index_variables": index_column_selector.value,
        "metric_variables": metric_columns_selector.value,
    })
    all_choices = []  # concatenate choices for parameters
    for key, entries in var_choices.items():
        if isinstance(entries, str):
            all_choices.append(entries)
        else:
            all_choices.extend([entry for entry in entries])
    return (all_choices,)


@app.cell
def _(all_choices, df, var_choices):
    from oss_app.utils import fix_column_names, fix_name

    # apply pre-filtering from selections above
    columns_included = df.columns if var_choices == {} else all_choices
    prefiltered_df = df[columns_included].copy()
    fix_column_names(prefiltered_df)
    return fix_name, prefiltered_df


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## <span style='color:lightcoral'>Filters
    ... Desc for grouping
    """
    )
    return


@app.cell
def _(loaded_params, mo):
    def format_filters(filters: dict):

        where_clause = [
            {"column_id": column, "operator": "equals", "value": value}
            for column, value in filters.items()
        ]
        filter_transforms = []
        if where_clause:  # if filters present
            filter_transforms.append({
                "type": "filter_rows",
                "operation": "keep_rows",
                "where": where_clause
            })
        return filter_transforms

    # load filters button
    load_filters_btn = mo.ui.run_button(
        label='Load saved filters', disabled=not loaded_params())

    return format_filters, load_filters_btn


@app.cell
def _(format_filters, loaded_params, previous_params):
    def load_filters(df, filter_params):
        import operator
        from functools import reduce
        df_next = df.copy()
        # combine column==value paired filters
        filters = reduce(operator.and_, (df_next[column].eq(
            value) for column, value in filter_params.items()))
        return df_next[filters]

    if loaded_params():
        loaded_transforms = format_filters(previous_params['filters'])
    return (loaded_transforms,)


@app.cell
def _(mo):
    confirm_filters = mo.ui.run_button(label='Confirm filters')
    return (confirm_filters,)


@app.cell
def _(mo):
    mo.output.append(mo.md(f'''### Click to transform dataset 
             For example, to select only female subjects or a specific experimental group, or both:<br>
             &emsp; click on transform and choose "Filter Rows", then click on "+ Add".'''))
    return


@app.cell
def _(mo):
    filters_loaded, set_filters_loaded = mo.state(False)
    return filters_loaded, set_filters_loaded


@app.cell
def _(load_filters_btn, set_filters_loaded):
    if load_filters_btn.value:
        set_filters_loaded(True)
    return


@app.cell
def _(filters_loaded, load_filters_btn, loaded_transforms, mo, prefiltered_df):
    load_filters_btn

    filter_ui = mo.ui.dataframe(prefiltered_df)

    if filters_loaded():

        filter_ui._Initialized = False
        preloaded_args = filter_ui._args
        preloaded_args.initial_value["transforms"] = loaded_transforms
        filter_ui._initialize(preloaded_args)
        filter_ui._Initialized = True
    return (filter_ui,)


@app.cell
def _(filter_ui, mo):
    mo.lazy(filter_ui)  # lazy allows for updates to reinitialize filter ui
    return


@app.cell
def _(load_filters_btn, mo):

    mo.output.append(load_filters_btn)

    if load_filters_btn.value:  # if loading saved filters
        with mo.redirect_stdout():
            print(
                '''Applied loaded filters... Changes might not appear in the dataframe preview.''')
    return


@app.cell
def _(confirm_filters, mo):
    mo.output.append(confirm_filters.right())
    return


@app.cell
def _(mo):
    mo.output.append(mo.md('<br>'))
    return


@app.cell
def _(confirm_filters, filter_ui, mo):
    mo.stop(confirm_filters.value is None)
    ui_transforms = filter_ui._last_transforms.transforms 
    return (ui_transforms,)


@app.cell
def _(ui_transforms):
    # retrieve filters applied
    if ui_transforms != []:  # transformations made or loaded
        filter_type = ui_transforms[0].type.value
        filter_operator = ui_transforms[0].operation
        final_transforms = {}

        transforms = ui_transforms
        if transforms != []:
            for transform in transforms:
                if transform.type.value == 'filter_rows':
                    for filter in transform.where:
                        col = filter.column_id
                        val = filter.value
                        final_transforms.update({col: val})
    else:
        final_transforms = {}
    return (final_transforms,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## <span style='color:lightcoral;font-family:arial'> Color selection
    <span style='font-family:arial'>Choose colormaps and colors used for plotting.
    """
    )
    return


@app.cell
def _(
    confirm_filters,
    fix_name,
    grouping_variable_selector,
    loaded_params,
    make_categorical,
    mo,
    plt,
    prefiltered_df,
    submit_choices,
):
    mo.stop((submit_choices.value is False) and (
        confirm_filters.value is False) and (not loaded_params()))
    _, labels = make_categorical(
        prefiltered_df[fix_name(grouping_variable_selector.value[0])]).items()
    group1, group2 = list(labels[1])

    def colormap_dropdn(x, y): return mo.ui.dropdown(
        label=x, options=plt.colormaps(), value=y, searchable=True)

    # default_colors = colormap_to_hex('tab10', 10)[:2]
    default_colors = ["#e45756", "#4c78a8"]
    def solidcolor_input(x, y): return mo.ui.text(
        placeholder="Color name or hex code here", label=x, value=y, debounce=True)

    group1_cmap = colormap_dropdn(f'"{group1}" colormap: ', 'viridis')
    group1_solid = solidcolor_input(
        f'"{group1}" solid color: ', default_colors[0])
    group2_cmap = colormap_dropdn(f'"{group2}" colormap: ', 'plasma')
    group2_solid = solidcolor_input(
        f'"{group2}" solid color: ', default_colors[1])
    return (
        group1,
        group1_cmap,
        group1_solid,
        group2,
        group2_cmap,
        group2_solid,
        labels,
    )


@app.cell
def _(labels, mo):
    mo.md(
        f"""
    #### <span style='font-family:arial'>Color (group) labels
    {labels[1][0]}, {labels[1][1]}
    """
    )
    return


@app.cell
def _(group1_cmap, group1_solid, group2_cmap, group2_solid, mo):
    color_choices = mo.ui.array([
        mo.md("{g1_cmap}, {g1_solid}").batch(
            g1_cmap=group1_cmap, g1_solid=group1_solid),
        mo.md("{g2_cmap}, {g2_solid}").batch(
            g2_cmap=group2_cmap, g2_solid=group2_solid),
    ], label='Color selections').form()

    color_choices
    return (color_choices,)


@app.cell
def _(color_choices, group1, group2, mo, show_color, show_colormap):
    if not color_choices.value:
        group1_colors = None
        group2_colors = None
    else:
        group1_colors = color_choices.value[0]
        group2_colors = color_choices.value[1]

    mo.stop(color_choices.value is None, mo.md(
        "###<span style='font-family:arial'>No colors chosen yet"))
    mo.output.append(
        mo.vstack([
            mo.md("###<span style='font-family:arial'>Color choices"),
            mo.hstack([
                mo.md(f"<span style='font-family:arial'>{group1}</span>"),
                show_colormap(group1_colors['g1_cmap']
                              ) if group1_colors else mo.md(""),
                show_color(group1_colors['g1_solid']
                           ) if group1_colors else mo.md(""),
            ], justify='start', align='center'),

            mo.hstack([
                mo.md(f"<span style='font-family:arial'>{group2}</span>"),
                show_colormap(group2_colors['g2_cmap']
                              ) if group2_colors else mo.md(""),
                show_color(group2_colors['g2_solid']
                           ) if group2_colors else mo.md(""),
            ], justify='start', align='center')
        ], gap=0)
    )
    return group1_colors, group2_colors


@app.cell
def _(confirm_filters, mo, submit_choices):
    mo.stop(submit_choices.value is False and confirm_filters.value is False)

    return


@app.cell
def _(
    filename,
    filepath,
    final_transforms,
    group1_colors,
    group2_colors,
    mo,
    var_choices,
):
    # update parameters output with filename and filters
    params_output = dict(
        filename=filename,
        filepath=str(filepath),
        **var_choices,
        filters=final_transforms,
        colors=[group1_colors, group2_colors]
    )

    # button to save parameters
    save_params = mo.ui.run_button(label="Save parameters", kind='warn')

    return params_output, save_params


@app.cell
def _(display_selections_markdown, mo, params_output, save_params, save_path):
    from oss_app.utils import save_parameters

    # display final selections
    mo.output.append(mo.vstack([
        display_selections_markdown(params_output),
        save_params.right()
    ]))

    if save_params.value:
        _file_out_path = save_parameters(params_output, save_path)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    <br>
    #  <span style='color:coral'> Analysis outputs
    * ...
    """
    )
    return


@app.cell
def _(mo, var_choices):
    # Prevent progression if choices not made yet for columns and filtering.
    mo.stop(var_choices == {} or var_choices is None, mo.md(
        "### Confirm variables, metrics and filtering to continue..."))

    return


@app.cell
def _(var_choices):
    var_choices
    return


@app.cell
def _(ds, filter_ui, var_choices):

    # initialize dataset object for processing
    data = ds.Dataset(
        raw_df=filter_ui.value,
        prefiltered=True,  # will attempt to filter based in parameter entries otherwise
        parameters=var_choices
    )
    data.initialize()  # sets filtered dataset, sets indices & metrics and corrects columns

    # example creating new metric
    # data.create_metric_from_operation(
    #     'interaction_center_ratio',
    #     'interaction : head time (s)', 'center : time (s)',
    #     operation=lambda a,b: a/b
    # )
    data.scale_metrics(per_group=False)
    data.calculate_si()
    data
    return (data,)


@app.cell
def _(data, mo):
    mo.vstack([
        mo.md("""### Preview datasets  
        Scaled metrics for each subject should add up to index score."""),
        mo.ui.tabs({
            # "Raw Data Preview": mo.plain(data.raw_df.head(3)),
            "Filtered Data Preview": mo.plain(data.filtered_df.head(3)),
            "Scaled Data Preview": mo.plain(data.scaled_df.head(3))
        }),
        mo.md("<br>")
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## <h2 style='font-family:arial; color:lightcoral'> Index score plots</h2>
    
    <span style='color:tomato;font-weight:bold;font-size:18px'>Population distributions</span>  
    <span style='font-size:12'>Plots comparing normalized distributions of index scores across group samples, either a single metric comparison or all selected metrics.</span>

    <span style='color:tomato;font-weight:bold;font-size:18px'>Individual distributions and correlations</span>   *- [WIP not implemented]*  
    Scatter plots show distributions over index scores within and between groups, along with distributions over correlations between index scores and each selected metric.


    <span style='color:tomato;font-weight:bold;font-size:18px'>PCA of index scores</span>  
    PCA biplot showing relative contributions to principal components of selected metrics composing the index score.
    """
    )
    return


@app.cell
def _(mo):
    # --- Marimo States ---
    # Initial range for the slider. This will be updated by the dropdown.
    get_range, set_range = mo.state([-12, 13])

    # State for the selected comparison metric
    get_compare_metric, set_compare_metric = mo.state('si_score')
    return get_range, set_compare_metric, set_range


@app.cell
def _(data, get_range, mo, np, set_compare_metric, set_range):

    # Define UI for rangex
    rangex_slider = mo.ui.range_slider(
        label="X-axis range",
        start=-20, stop=20, step=0.5,
        show_value=False, debounce=True,
        value=get_range(),    # The current value of the slider comes from the state
        on_change=set_range,  # When the slider is manually moved, update the state
        full_width=False
    )

    rangey_slider = mo.ui.slider(
        start=0, stop=100, step=5,
        orientation='vertical',
        value=30, debounce=True, full_width=True
    )

    # Define function for updating range
    def on_metric_change(new_metric_value):
        global data
        # Update the selected metric state
        set_compare_metric(new_metric_value)

        df = data.scaled_df.copy()
        # --- Logic to automatically set slider range based on new_metric_value ---
        if new_metric_value == 'si_score':
            # Specific range for 'SI_scores' if it's not a direct column
            set_range([-12, 13])  # Or calculate if SI_scores is derived
        elif new_metric_value in df.columns:
            # Get min/max from the actual data for the selected column
            min_val = df[new_metric_value].min()
            max_val = df[new_metric_value].max()

            # Add a little padding for better visualization
            padding = (max_val - min_val) * 0.3
            padding = round(padding)
            # Ensure padding doesn't result in inverted range or too small
            if min_val == max_val:
                min_val -= 1
                max_val += 1

            # Update the range state, which will automatically update rangex_slider.value
            set_range([np.floor(min_val, dtype=float) - padding,
                      np.ceil(max_val, dtype=float) + padding])
        else:
            # Fallback default
            # A reasonable default if no data-driven range is found
            set_range([-15, 15])

    # Define UI for compare_metric
    # Combine SI_scores with other metrics_included
    cleaned_metrics = data.metric_variables
    available_compare_metrics = ['si_score'] + cleaned_metrics

    compare_metric_selector = mo.ui.dropdown(
        options=available_compare_metrics,
        value='si_score',
        label="Select Metric for Comparison",
        on_change=on_metric_change  # Call custom handler
    )

    save_distplot_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_distplots_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_scatter_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_correlations_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_pca_biplot_button = mo.ui.run_button(label='Save plot', kind='warn')
    save_pca_matrix_button = mo.ui.run_button(label='Save plot', kind='warn')

    return (
        cleaned_metrics,
        compare_metric_selector,
        rangex_slider,
        rangey_slider,
        save_distplot_button,
        save_distplots_button,
        save_pca_biplot_button,
    )


@app.cell
def _(
    compare_metric_selector,
    data,
    get_range,
    group1_colors,
    group2_colors,
    mo,
    rangex_slider,
    rangey_slider,
    save_distplot_button,
    save_path,
):
    from oss_app.plotting import compare_dists_altair

    SI_plot_params = dict(
        set_size_params=(3.5, 3.5),
        hide_text=False,
    )

    def plot_distribution(compare_metric: str, max_y: int = None, rangex: list = []):

        if max_y is None:
            max_y = rangey_slider.value
        if rangex == []:
            rangex = get_range()

        # Generate the Altair plot using the selected UI values
        altplot_interactive = compare_dists_altair(
            data.scaled_df,
            # compare_metric=compare_metric_selector.value,
            compare_metric=compare_metric,
            group_variable=data.grouping_variable,
            filters={},
            max_y=max_y,          # y axis max
            rangex=rangex,  # x axis range
            user_colors=[group1_colors['g1_solid'], group2_colors['g2_solid']],
            user_labels=None,  # or eg, ['delayed', 'immediate']
            legend=True,       # add legends
            set_size_params=(3.5, 3.5),  # size of plot,
            hide_text=False,    # hide all text in plot
            alpha1=0.5,
            alpha2=0.5
        ).properties(width=350, height=350).interactive()
        return altplot_interactive

    def dist_plot_layout(compare_metric: str = ''):
        if compare_metric == '':
            compare_metric = compare_metric_selector.value

        # Define and display the UI elements and interactive Altair chart
        right_content = mo.vstack([
            mo.md(
                f'Y-axis max: <br>{rangey_slider.value}').style({'text-align': 'center'}),
            rangey_slider.center()
        ],  align='center', gap=0.5).style({'width': '80px', 'align-content': 'center'})

        xlim = [float(x) for x in get_range()]  # retrieve x range values
        dist_layout = mo.vstack([
            compare_metric_selector,
            mo.hstack([
                rangex_slider,
                mo.md(f'{xlim[0]} :::::&nbsp;{xlim[1]}')
            ], justify='start', align='end',  gap=1).center(),
            mo.hstack([
                # mo.ui.altair_chart(altplot_interactive),
                distribution_plot := plot_distribution(compare_metric),
                right_content,
                mo.md("")
            ], justify='start', align='stretch', gap=0)
        ], align='stretch', gap=1)
        return dist_layout, distribution_plot

    with mo.capture_stdout() as _buffer:
        dist_layout, dist_plot = dist_plot_layout()

    def save_plot(altair_plot, plot_filepath, format: str = 'png'):
        altair_plot.save(plot_filepath, format=format)
        print(f'\nSaved plot to "{plot_filepath}".')

    # Layout
    mo.output.append(mo.vstack([
        mo.md("### <span style='font-family:arial;border:2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:Single metric distribution comparison"),
        mo.md(f"""<span style='font-family:arial;font-size:16px'>
            Compare normalized population distributions between groups for a given metric (default: index score).<br></span>
            <br>
            """).style(color="white"),
        dist_layout,
        save_distplot_button.right(),
    ]))

    _console_logs = ['<br>'+line for line in _buffer.getvalue().split('\n')]
    _formatted_logs = [
        f"<span style='display:block;font-size:13px;line-height:8px;font-family:monospace'>{log}" for log in _console_logs]

    mo.output.append(
        mo.md(
            f"""/// details | Console outputs
                type: info 
            {mo.as_html(mo.md('<br>'.join(_formatted_logs)))}
            ///"""
        ))

    if save_distplot_button.value:
        plot_filepath = save_path / \
            f'distplot.png'  # TODO: change label
        with mo.redirect_stdout():
            save_plot(dist_plot, plot_filepath)

    return dist_plot, plot_distribution, save_plot


@app.cell
def _(
    alt,
    cleaned_metrics,
    data,
    mo,
    plot_distribution,
    save_distplots_button,
    save_path,
    save_plot,
):

    dist_plots = []
    with mo.capture_stdout() as _buffer:
        for metric in cleaned_metrics:
            plot_data = data.scaled_df[metric]
            xmin = plot_data.min().round()-2
            xmax = plot_data.max().round()+2

            mean = plot_data.mean()
            std_dev = plot_data.std()
            max_y = 100
            dist_plots.append(plot_distribution(
                metric, max_y=max_y, rangex=[xmin, xmax]))

    # concatenate the list of charts
    final_chart = alt.hconcat(*dist_plots).resolve_scale(y='independent')

    # Layout
    mo.output.append(mo.vstack([
        mo.md("### <span style='font-family:arial;border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:Distribution comparisons for chosen metrics"),
        mo.md(f"""<span style='font-family:arial;font-size:16px'>
            Compare normalized population distributions between groups for all selected metrics.<br></span>
            <br>
            """).style(color="white"),
        final_chart,
        save_distplots_button.right()
    ]))

    _console_logs = [line for line in _buffer.getvalue().split('\n')[1:]]
    _formatted_logs = [
        f"<span style='display:block;font-size:13px;line-height:1.25em;font-family:monospace'>{log}" for log in _console_logs]

    mo.output.append(
        mo.md(
            f"""/// details | Console outputs
                type: info 
            {mo.as_html(mo.md('<br>'.join(_formatted_logs)))}
            ///"""
        ))

    if save_distplots_button.value:
        dist_plots_filepath = save_path / \
            'distplots.png'  # TODO: change label
        with mo.redirect_stdout():
            save_plot(final_chart, dist_plots_filepath)
    return


@app.cell
def _(mo):

    # Layout
    mo.output.append(mo.vstack([
        mo.md("### <span style='font-family:arial;border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:<span style='color: crimson'>(Not implemented yet!)</span> Scatterplot metrics"),
        mo.md(f"""<span style='font-family:arial;font-size:16px'>
            Distribution scatterplots of individual data points across selected metrics, each data point colored by sample index score. Can use z-scored or raw metric values. 
            <br></span>
            <br>
            """),
        # mo.md("""**Example**<br>
        # z-scored values    
        # <img src="public/si_scatters_zscore.png" width="500" />
        # raw values  
        # <img src="public/si_scatters_raw.png" width="500" />
        # """),
        # save_scatter_button.right()
    ]))
    return


@app.cell
def _(mo):

    # Layout
    mo.output.append(mo.vstack([
        mo.md("### <span style='font-family:arial;border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small:<span style='color: crimson'>(Not implemented yet!)</span> Correlation scatter metrics"),
        mo.md(f"""<span style='font-family:arial;font-size:16px'>
            Correlation scatterplots with individual distributions for each selected metric against their index score, also colored by index score. Can use z-scored or raw metric values.
            <br></span>
            <br>
            """),
        # mo.md("""**Example**<br>
        # z-scored values  
        # ![example](public/labels.png) 
        # ![example](public/si_correls_zscore.png)  
        # raw values  
        # ![example](public/labels.png) 
        # ![example](public/si_correls_raw.png) 
        # """),
        # save_scatter_button.right()
    ]))
    return


@app.cell
def _(alt, data, dist_plot, mo, save_path, save_pca_biplot_button, save_plot):
    from oss_app.plotting import do_pca, pca_biplot_altair, set_global_font
    alt.themes.register("arial_font", set_global_font)
    alt.themes.enable("arial_font")

    def plot_pca_biplot(df):#compare_metric: str, max_y: int = None, rangex: list = []):
    
        pca_metrics = [m for m in df.metric_variables if m != 'si_score']
        pca_data = df.scaled_df
    
        # Generate the Altair plot using the selected UI values
        altplot_interactive = pca_biplot_altair(
            df,
            metrics_included=pca_metrics,
            labels="",
            pca_inputs=None,
            n_comp=3,
            mapping=0,
            pcs=None,
            colorset=None,
            hide_text=False
        )
        return altplot_interactive

    with mo.capture_stdout() as _buffer:
        biplot, legend = plot_pca_biplot(data)

    # Layout
    mo.output.append(mo.vstack([
        mo.md("### <span style='font-family:arial;border: 2px solid coral;padding:4px 5px;display:inline-block'>:arrow_down_small: PCA analysis"),
        mo.md(f"""<span style='font-family:arial;font-size:16px'>
            PCA includes two plots:  
            - biplot of individual data points for scaled metrics, colored by index score  
            - correlation matrix of scaled metrics to each principal component - WIP
            <br></span>
            """).style(color="white"),
        mo.hstack([biplot, legend], justify='start'),
        # dist_layout,
        save_pca_biplot_button.right(),
    ]))

    _console_logs = ['<br>'+line for line in _buffer.getvalue().split('\n')]
    _formatted_logs = [
        f"<span style='display:block;font-size:13px;line-height:8px;font-family:monospace'>{log}" for log in _console_logs]

    mo.output.append(
        mo.md(
            f"""/// details | Console outputs
                type: info 
            {mo.as_html(mo.md('<br>'.join(_formatted_logs)))}
            ///"""
        ))

    if save_pca_biplot_button.value:
        _plot_filepath = save_path / \
            f'pca_biplot.png'  # TODO: change label
        with mo.redirect_stdout():
            save_plot(dist_plot, _plot_filepath)
    return


if __name__ == "__main__":
    app.run()
