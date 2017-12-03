#!/usr/bin/env python
"""Integration testing."""

# System
import os
import logging
import time

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

DATA_SET_PATH = 'numerai_datasets'

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger('integration_test')


def main():
    # when running on circleci, set the vars in the project settings
    email = os.environ.get('NUMERAPI_EMAIL')
    password = os.environ.get('NUMERAPI_PASS')

    if email is None or password is None:
        raise RuntimeError('set the NUMERAPI_EMAIL and NUMERAPI_PASS environment variables first')

    napi = NumerAPI()
    napi.credentials = (email, password)

    test_csv = "test_csv"
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

    valid = tournament_data["data_type"] == "validation"
    test = tournament_data["data_type"] != "validation"

    x_pv, ids_v = x_prediction[valid], ids[valid]
    x_pt, ids_t = x_prediction[test], ids[test]

    clfs = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        # GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        XGBClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100, nthread=-1),
        #KNeighborsClassifier(3, n_jobs=1),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1, hidden_layer_sizes=(100, 100)),
        # AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(tol=1.0e-3),
        LogisticRegression(n_jobs=-1, solver='sag')
    ]

    for clf in clfs:
        before = time.time()
        clf_str = str(clf).split("(")[0]
        clf.fit(X, Y)
        time_taken = '%.2fs' % (time.time()-before)
        logger.info('fit() took %s%s (%s)' % (time_taken, ' '*(9-len(time_taken)), clf_str))

    for clf in clfs:
        clf_str = str(clf).split("(")[0]

        before = time.time()
        y_prediction = clf.predict_proba(x_prediction)
        after = time.time()

        results = y_prediction[:, 1]
        results_df = pd.DataFrame(data={'prediction': results})
        joined = pd.DataFrame(ids).join(results_df)

        out = os.path.join(test_csv, "{}-legit.csv".format(clf_str))
        time_taken = '%.2fs' % (after-before)
        logger.info('predict_proba() took %s%s (%s)' % (time_taken, ' '*(9-len(time_taken)), out))

        # Save the predictions out to a CSV file
        joined.to_csv(out, index=False)

        # TODO: when api fixed, add the submission_id to  exc pool, async checks status to; if any fails: sys.exit(1)
        napi.upload_prediction(out)
        # input("Both concordance and originality should pass. Press enter to continue...")

    before = time.time()
    # TODO: use nose and parameterize this
    for i, clf1 in enumerate(clfs):
        for j, clf2 in enumerate(clfs):
            if i == j:
                continue

            before_one_mix = time.time()
            y_pv = clf1.predict_proba(x_pv)[:, 1]
            valid_df = pd.DataFrame(ids_v).join(pd.DataFrame(data={'prediction': y_pv}))

            y_pt = clf2.predict_proba(x_pt)[:, 1]
            test_df = pd.DataFrame(ids_t).join(pd.DataFrame(data={'prediction': y_pt}))

            mix = pd.concat([valid_df, test_df])

            out = os.path.join(test_csv, "{}-{}-mix.csv".format(str(clf1).split("(")[0], str(clf2).split("(")[0]))
            mix.to_csv(out, index=False)

            # TODO: when api fixed, add the submission_id to  exc pool, async checks status to; if pass: sys.exit(1)
            response = napi.upload_prediction(out)
            # input("Concordance should fail. Press enter to continue...")
            time_taken = '%.2fs' % (time.time()-before_one_mix)
            logger.info(
                'upload_prediction() took %s%s (%s, %s)' % (time_taken, ' '*(10-len(time_taken)), str(response), out))

    logger.info('all mix clfs predict_proba() took %.2fs' % (time.time()-before))


if __name__ == '__main__':
    main()
