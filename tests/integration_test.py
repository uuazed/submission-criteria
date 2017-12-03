#!/usr/bin/env python
"""Integration testing."""

# System
import os
import logging
import time

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # , AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost.sklearn import XGBClassifier

# First Party
from testing_api import NumerAPI

DATA_SET_PATH = 'tests/numerai_datasets'
test_csv = "tests/test_csv"
clf_n_jobs = 2

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger('integration_test')
upload_executor = ThreadPoolExecutor(max_workers=10)
clf_executor = ThreadPoolExecutor(max_workers=clf_n_jobs)


def main():
    # when running on circleci, set the vars in the project settings
    email = os.environ.get('NUMERAPI_EMAIL')
    password = os.environ.get('NUMERAPI_PASS')

    if email is None or password is None:
        raise RuntimeError('set the NUMERAPI_EMAIL and NUMERAPI_PASS environment variables first')

    napi = NumerAPI()
    napi.credentials = (email, password)

    if not os.path.exists(test_csv):
        os.makedirs(test_csv)

    if not os.path.exists(DATA_SET_PATH):
        logger.info("Downloading the current dataset...")
        os.makedirs(DATA_SET_PATH)
        napi.download_current_dataset(dest_path=DATA_SET_PATH, unzip=True)
    else:
        logger.info("Found old data to use.")

    training_data = pd.read_csv('%s/numerai_training_data.csv' % DATA_SET_PATH, header=0)
    tournament_data = pd.read_csv('%s/numerai_tournament_data.csv' % DATA_SET_PATH, header=0)

    features = [f for f in list(training_data) if "feature" in f]
    features = features[:len(features)//2]  # just use half, speed things up a bit
    X, Y = training_data[features], training_data["target"]

    x_prediction = tournament_data[features]
    ids = tournament_data["id"]

    clfs = [
        RandomForestClassifier(
            n_estimators=15, max_features=1, max_depth=2, n_jobs=1, criterion='entropy', random_state=42),
        # GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        XGBClassifier(learning_rate=0.1, subsample=0.4, max_depth=2, n_estimators=20, nthread=1, seed=42),
        # KNeighborsClassifier(3, n_jobs=1),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        MLPClassifier(alpha=1, hidden_layer_sizes=(25, 25), random_state=42),
        # AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(tol=1.0e-3),
        LogisticRegression(n_jobs=2, solver='sag', C=1, tol=1e-2, random_state=42, max_iter=50)
    ]

    before = time.time()
    fit_all(clfs, X, Y)
    logger.info('all clfs fit() took %.2fs' % (time.time()-before))

    before = time.time()
    uploads_wait_for_legit = predict_and_upload_legit(napi, clfs, x_prediction, ids)
    logger.info('all legit clfs predict_proba() took %.2fs' % (time.time()-before))

    before = time.time()
    uploads_wait_for_mix = predict_and_upload_mix(napi, clfs, tournament_data, x_prediction, ids)
    logger.info('all mix clfs predict_proba() took %.2fs' % (time.time()-before))

    before = time.time()
    for f in futures.as_completed(uploads_wait_for_legit):
        pass  # TODO: logger.info('future done, result: %s' % str(f))
    logger.info('await legit uploads took %.2fs' % (time.time() - before))

    before = time.time()
    for f in futures.as_completed(uploads_wait_for_mix):
        pass  # TODO: logger.info('future done, result: %s' % str(f))
    logger.info('await mix uploads took %.2fs' % (time.time() - before))


def fit_all(clfs: list, X, Y):
    wait_for = list()

    for clf in clfs:
        wait_for.append(clf_executor.submit(fit_clf, X, Y, clf))

    before = time.time()
    for _ in futures.as_completed(wait_for):
        pass
    logger.info('await fitting took %.2fs' % (time.time()-before))


def fit_clf(X, Y, clf):
    before = time.time()
    clf_str = str(clf).split("(")[0]
    clf.fit(X, Y)
    time_taken = '%.2fs' % (time.time()-before)
    logger.info('fit() took %s%s (%s)' % (time_taken, ' '*(9-len(time_taken)), clf_str))
    return clf_str


def predict_and_upload_legit(napi, clfs: list, x_prediction, ids):
    wait_for = list()
    upload_wait_for = list()

    for clf in clfs:
        wait_for.append(clf_executor.submit(predict_and_upload_one_legit, upload_wait_for, napi, clf, x_prediction, ids))

    before = time.time()
    for f in futures.as_completed(wait_for):
        pass  # TODO: logger.info('future done, result: %s' % str(f))
    logger.info('await legit predictions took %.2fs' % (time.time()-before))

    return upload_wait_for


def predict_and_upload_one_legit(upload_wait_for: list, napi, clf, x_prediction, ids):
    clf_str = str(clf).split("(")[0]

    before = time.time()
    y_prediction = clf.predict_proba(x_prediction)
    after = time.time()

    out = os.path.join(test_csv, "{}-legit.csv".format(clf_str))
    time_taken = '%.2fs' % (after - before)
    logger.info('predict_proba() took %s%s (%s)' % (time_taken, ' ' * (9 - len(time_taken)), out))

    upload_wait_for.append(upload_executor.submit(upload_one_legit, y_prediction, ids, out, napi))


def upload_one_legit(y_prediction, ids, out: str, napi):
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'prediction': results})
    joined = pd.DataFrame(ids).join(results_df)

    # Save the predictions out to a CSV file
    joined.to_csv(out, index=False)

    # TODO: when api fixed, add the submission_id to  exc pool, async checks status to; if any fails: sys.exit(1)
    # input("Both concordance and originality should pass. Press enter to continue...")
    napi.upload_prediction(out)


def predict_and_upload_mix(napi, clfs: list, tournament_data: pd.DataFrame, x_prediction, ids):
    valid = tournament_data["data_type"] == "validation"
    test = tournament_data["data_type"] != "validation"

    x_pv, ids_v = x_prediction[valid], ids[valid]
    x_pt, ids_t = x_prediction[test], ids[test]

    wait_for = list()
    uploads_wait_for = list()

    for i, clf1 in enumerate(clfs[:len(clfs)//2]):
        for j, clf2 in enumerate(clfs[:len(clfs)//2]):
            if i == j:
                continue

            wait_for.append(clf_executor.submit(predict_and_upload_one_mix, napi, uploads_wait_for, clf1, clf2, x_pv, x_pt, ids_v, ids_t))

    before = time.time()
    for f in futures.as_completed(wait_for):
        pass  # TODO: logger.info('future done, result: %s' % str(f))
    logger.info('await mix predictions took %.2fs' % (time.time()-before))
    return uploads_wait_for


def predict_and_upload_one_mix(napi, uploads_wait_for: list, clf1, clf2, x_pv, x_pt, ids_v, ids_t) -> None:
    before_one_mix = time.time()
    y_pv = clf1.predict_proba(x_pv)[:, 1]
    y_pt = clf2.predict_proba(x_pt)[:, 1]

    out = os.path.join(test_csv, "{}-{}-mix.csv".format(str(clf1).split("(")[0], str(clf2).split("(")[0]))
    time_taken = '%.2fs' % (time.time() - before_one_mix)
    logger.info(
        'pred mix took  %s%s (%s)' % (time_taken, ' ' * (10 - len(time_taken)), out))

    uploads_wait_for.append(upload_executor.submit(upload_one_mix, napi, out, ids_v, ids_t, y_pt, y_pv))


def upload_one_mix(napi, out, ids_v, ids_t, y_pt, y_pv):
    valid_df = pd.DataFrame(ids_v).join(pd.DataFrame(data={'prediction': y_pv}))
    test_df = pd.DataFrame(ids_t).join(pd.DataFrame(data={'prediction': y_pt}))

    before_csv_write = time.time()
    mix = pd.concat([valid_df, test_df])
    mix.to_csv(out, index=False)
    time_taken = '%.2fs' % (time.time() - before_csv_write)
    logger.info('write csv took %s%s (%s)' % (time_taken, ' ' * (10 - len(time_taken)), out))

    # TODO: when api fixed, add the submission_id to  exc pool, async checks status to; if pass: sys.exit(1)
    before_upload = time.time()
    response = napi.upload_prediction(out)
    time_taken = '%.2fs' % (time.time() - before_upload)
    # logger.info('upload took   %s%s (%s, %s)' % (time_taken, ' '*(10-len(time_taken)), str(response), file_to_upload))
    # input("Concordance should fail. Press enter to continue...")


if __name__ == '__main__':
    main()
