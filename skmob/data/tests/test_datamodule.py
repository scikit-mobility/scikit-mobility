import pytest

from skmob.data.load import list_datasets, load_dataset

#  test 1: CORRECT arguments, no ERRORS expected (#test: 4)

"""
@pytest.mark.parametrize(
    "dataset_names", ["flow_foursquare_nyc", "foursquare_nyc", "nyc_boundaries", "parking_san_francisco"]
)
def test_loading_existing_dataset(dataset_names):

    data = load_dataset(dataset_names)

    assert type(data) in [TrajDataFrame, FlowDataFrame, GeoDataFrame, DataFrame]
"""

# test 2: INCORRECT dataset name, ERRORS expected (#test: 2)


@pytest.mark.parametrize("dataset_names", ["thisdatasetdoesnotexist", ""])
@pytest.mark.xfail(raises=ValueError)
def test_loading_non_existing_dataset(dataset_names):

    load_dataset(dataset_names)


"""
# test 3: drop_colums with CORRECT arguments, no ERRORS expected (#test: 2)
@pytest.mark.parametrize("dataset_names", ["foursquare_nyc"])
@pytest.mark.parametrize("drop_columns", [True, False])
def test_loading_drop_cols(dataset_names, drop_columns):

    data = load_dataset(dataset_names, drop_columns=drop_columns)

    assert isinstance(data, TrajDataFrame)
"""

# test 4: WRONG arguments type, ERRORS expected (#test: 16)


@pytest.mark.parametrize("dataset_names", ["foursquare_nyc"])
@pytest.mark.parametrize("drop_columns", [True, "False"])
@pytest.mark.parametrize("auth", [("a", 2), (), (1, 2, 3), ("user", "psw")])
@pytest.mark.parametrize("show_progress", ["True", "False"])
@pytest.mark.xfail(raises=ValueError)
def test_loading_wrong_types(dataset_names, drop_columns, auth, show_progress):

    load_dataset(dataset_names, drop_columns=drop_columns, auth=auth, show_progress=show_progress)


# test 5: CORRECT arguments, NO ERRORS expected (#test: 14)
@pytest.mark.parametrize("details", [True, False])
@pytest.mark.parametrize(
    "data_types", [None, [], ["shape"], ["trajectory"], ["flow"], ["auxiliar"], ["trajectory", "auxiliar"]]
)
def test_list_datasets(details, data_types):

    d_list = list_datasets(details=details, data_types=data_types)

    if details:
        assert isinstance(d_list, dict)
    else:
        assert isinstance(d_list, list)


# test 6: WRONG arguments for details, ERRORS expected (#test: 4)
@pytest.mark.parametrize("details", ["True", None, "", []])
@pytest.mark.xfail(raises=ValueError)
def test_list_datasets_wrong_details(details):

    list_datasets(details=details)


# test 7: WRONG arguments for data_types, ERRORS expected (#test: 4)
@pytest.mark.parametrize("data_types", [[1], "True", None, ""])
@pytest.mark.xfail(raises=ValueError)
def test_list_datasets_wrong_data_types(data_types):

    list_datasets(data_types=data_types)


"""
def test_correctness_urls():

    all_datasets = list_datasets(details=True)

    for data_k in all_datasets:
        url_2_check = all_datasets[data_k]["url"]
        try:
            requests.head(url_2_check)
        except ConnectionError:
            assert 1 == 0

    assert 1 == 1
"""
