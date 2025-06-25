import os


def set_model(model="FX_neutral"):
    from factor_investing.nodes.dao.folder_dao import empty_cache

    empty_cache()

    os.environ["ALPHAGP_START_TS"] = ""
    os.environ["ALPHAGP_END_TS"] = ""
    os.environ["ALPHAGP_MODEL"] = model


def set_test_data_path():
    test_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        )
    )

    database_path = test_dir + os.sep + rf"data/ddbb"
    os.environ['LAMBDA_DATA_PATH'] = database_path
    os.environ['LAMBDA_INPUT_PATH'] = rf"{database_path}/input_models"
    return database_path


def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper
