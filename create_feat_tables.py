"""
Contains all code responsible for generating the feature-engineered datasets from the RelBench User Study.

Repository of the original code: https://github.com/snap-stanford/relbench-user-study
"""

import os

import duckdb
import pandas as pd
from components.external.userstudy.utils import db_setup, render_jinja_sql
from pandarallel import pandarallel
from relbench.datasets import get_dataset
from relbench.tasks import get_task

pandarallel.initialize(progress_bar=True, nb_workers=10)


def from_sql(dataset: str, task: str):
    dataset_path = os.path.join("cache", "feat", dataset)
    os.makedirs(os.path.join(dataset_path, task), exist_ok=True)
    if not os.path.exists(os.path.join(dataset_path, "database.db")):
        print(f"Setting up db for {dataset}...")
        db_setup(dataset, os.path.join(dataset_path, "database.db"))

    sql_path = os.path.join(
        "components", "external", "userstudy", dataset.split("-")[1], task, "feats.sql"
    )
    conn = duckdb.connect(os.path.join(dataset_path, "database.db"))
    with open(sql_path, "r") as f:
        template = f.read()
    print(f"Querying for task {task}...")
    for s in ["train", "val", "test"]:
        query = render_jinja_sql(template, dict(set=s, subsample=0))
        conn.sql(query)
    train_df = conn.sql(f"select * from {task.replace('-', '_')}_train_feats").df()
    val_df = conn.sql(f"select * from {task.replace('-', '_')}_val_feats").df()
    test_df = conn.sql(f"select * from {task.replace('-', '_')}_test_feats").df()
    conn.close()

    print("Remerging...")
    rel_task = get_task(dataset, task, download=True)
    train_df = rel_task.get_table("train").df.merge(
        train_df, on=list([x for x in rel_task.get_table("train").df.columns if not (dataset=="rel-event" and x == "index")]), how="left"
    )
    val_df = rel_task.get_table("val").df.merge(
        val_df, on=list([x for x in rel_task.get_table("val").df.columns if not (dataset=="rel-event" and x == "index")]), how="left"
    )
    test_df = rel_task.get_table("test").df.merge(
        test_df, on=list([x for x in rel_task.get_table("test").df.columns if not (dataset=="rel-event" and x == "index")]), how="left"
    )

    print("Saving...")
    drop_cols = get_drop_cols(dataset, task)
    train_df.drop(drop_cols, axis=1).to_csv(
        os.path.join(dataset_path, task, "train.csv")
    )
    val_df.drop(drop_cols, axis=1).to_csv(os.path.join(dataset_path, task, "val.csv"))
    test_df.drop(drop_cols, axis=1).to_csv(os.path.join(dataset_path, task, "test.csv"))


def from_ipynb(dataset: str, task: str, function):
    dataset_path = os.path.join("cache", "feat", dataset)
    os.makedirs(os.path.join(dataset_path, task), exist_ok=True)
    rel_task = get_task(dataset, task, download=True)
    orig_train_df = rel_task.get_table("train").df
    orig_val_df = rel_task.get_table("val").df
    orig_test_df = rel_task.get_table("test").df

    print(f"Calculating features for task {task}...")
    train_df = function(orig_train_df)
    val_df = function(orig_val_df)
    test_df = function(orig_test_df)

    print("Remerging...")
    train_df = orig_train_df.merge(train_df, on=list(orig_train_df.columns), how="left")
    val_df = orig_val_df.merge(val_df, on=list(orig_val_df.columns), how="left")
    test_df = orig_test_df.merge(test_df, on=list(orig_test_df.columns), how="left")

    print("Saving...")
    drop_cols = get_drop_cols(dataset, task)
    train_df.drop(drop_cols, axis=1).to_csv(
        os.path.join(dataset_path, task, "train.csv")
    )
    val_df.drop(drop_cols, axis=1).to_csv(os.path.join(dataset_path, task, "val.csv"))
    test_df.drop(drop_cols, axis=1).to_csv(os.path.join(dataset_path, task, "test.csv"))


def get_drop_cols(dataset: str, task: str):
    if dataset != "rel-trial":
        return TASK_PARAMS[dataset + "-" + task]["identifier_cols"]
    else:
        if task == "site-success":
            return ["facility_id", "timestamp"]
        if task == "study-outcome":
            return ["nct_id", "timestamp"]


def main():
    sql_dict = {
        "rel-amazon": ["item-churn", "item-ltv", "user-churn", "user-ltv"],
        "rel-event": ["user-attendance", "user-ignore", "user-repeat"],
        "rel-f1": ["driver-dnf", "driver-position", "driver-top3"],
        "rel-hm": ["item-sales", "user-churn"],
        "rel-stack": ["post-votes", "user-badge", "user-engagement"],
    }
    for dataset, tasks in sql_dict.items():
        for task in tasks:
            from_sql(dataset, task)
        dataset_path = os.path.join("cache", "feat", dataset)
        os.remove(os.path.join(dataset_path, "database.db"))

    from_ipynb("rel-trial", "site-success", get_features_site_success)
    from_ipynb("rel-trial", "study-outcome", get_features_study_outcome)


# From userstudy ipynb
def get_features_site_success(orig_train_df):
    train_df = orig_train_df.copy()
    dataset = get_dataset(name="rel-trial", download=True)
    task = get_task("rel-trial", "site-success", download=True)
    tables = dataset.get_db().table_dict

    interventions_studies = tables["interventions_studies"].df
    conditions_studies = tables["conditions_studies"].df
    sponsors_studies = tables["sponsors_studies"].df
    facilities_studies = tables["facilities_studies"].df
    studies = tables["studies"].df
    designs = tables["designs"].df
    eligibilities = tables["eligibilities"].df
    sponsors = tables["sponsors"].df
    reported_event_totals = tables["reported_event_totals"].df

    outcomes = tables["outcomes"].df
    facilities = tables["facilities"].df
    outcome_analyses = tables["outcome_analyses"].df
    outcome_analyses = outcome_analyses.merge(
        outcomes[["id", "outcome_type"]], left_on="outcome_id", right_on="id"
    )

    def get_derived_features(facility_id, timestamp):
        try:
            history_ncts = facilities_studies[
                (facilities_studies.facility_id == facility_id)
                & (facilities_studies.date < timestamp)
            ].nct_id.unique()
            num_trials_conducted = len(history_ncts)
            num_conditions_conducted = len(
                conditions_studies[
                    conditions_studies.nct_id.isin(history_ncts)
                ].condition_id.unique()
            )
            num_interventions_conducted = len(
                interventions_studies[
                    interventions_studies.nct_id.isin(history_ncts)
                ].intervention_id.unique()
            )
            num_sponsors_conducted = len(
                sponsors_studies[
                    sponsors_studies.nct_id.isin(history_ncts)
                ].sponsor_id.unique()
            )

            studies_temp = studies[studies.nct_id.isin(history_ncts)]
            studies_temp["phase"] = studies_temp.phase.astype(str)
            num_phase1 = len(studies_temp[studies_temp.phase.str.contains("1")])
            num_phase2 = len(studies_temp[studies_temp.phase.str.contains("2")])
            num_phase3 = len(studies_temp[studies_temp.phase.str.contains("3")])
            num_phase4 = len(studies_temp[studies_temp.phase.str.contains("4")])

            outcome_temp = outcome_analyses[outcome_analyses.nct_id.isin(history_ncts)]
            outcome_temp = outcome_temp[
                (outcome_temp.p_value_modifier.isnull())
                | (outcome_temp.p_value_modifier != ">")
            ]
            outcome_temp = outcome_temp[
                (outcome_temp.p_value >= 0)
                & (outcome_temp.p_value <= 1)
                & (outcome_temp.outcome_type == "Primary")
            ]
            nct2p = outcome_temp.groupby("nct_id").p_value.min()
            if len(nct2p) == 0:
                avg_history_success_rate = 0
            else:
                avg_history_success_rate = sum(nct2p <= 0.05) / len(nct2p)

            event_temp = reported_event_totals[
                reported_event_totals.nct_id.isin(history_ncts)
            ]
            event_temp = event_temp[
                (event_temp.event_type == "serious")
                | (event_temp.event_type == "deaths")
            ]
            event_temp = event_temp[~event_temp.subjects_affected.isnull()]
            nct2event = event_temp.groupby("nct_id").subjects_affected.sum()

            if len(nct2event) == 0:
                avg_history_adverse_events = 0
            else:
                avg_history_adverse_events = sum(nct2event) / len(nct2event)

            max_historical_trial_enrollment = studies_temp.enrollment.max()
            avg_historical_trial_enrollment = studies_temp.enrollment.mean()

            operation_years = (
                studies_temp.start_date.max().year - studies_temp.start_date.min().year
            )
            years_since_last_trial_facility = (
                timestamp.year - studies_temp.start_date.max().year
            )

            nct2num_facility = dict(
                facilities_studies[(facilities_studies.nct_id.isin(history_ncts))]
                .groupby("nct_id")
                .facility_id.agg(len)
            )
            if len(nct2num_facility) == 0:
                in_multi_center_trial = 0
                num_multi_center_trial = 0
            else:
                if min(nct2num_facility.values()) > 1:
                    in_multi_center_trial = 1
                else:
                    in_multi_center_trial = 0
                num_multi_center_trial = sum(
                    [1 if j > 1 else 0 for i, j in nct2num_facility.items()]
                )

            return [
                num_trials_conducted,
                num_conditions_conducted,
                num_interventions_conducted,
                num_sponsors_conducted,
                num_phase1,
                num_phase2,
                num_phase3,
                num_phase4,
                avg_history_success_rate,
                avg_history_adverse_events,
                max_historical_trial_enrollment,
                avg_historical_trial_enrollment,
                operation_years,
                years_since_last_trial_facility,
                in_multi_center_trial,
                num_multi_center_trial,
            ]
        except Exception as e:
            print(e)

    train_df["derived_features"] = train_df.parallel_apply(
        lambda x: get_derived_features(x.facility_id, x.timestamp), axis=1
    )

    facility_to_merge_col = []
    for country in facilities.country.value_counts()[:10].keys():
        facilities["in_" + country.lower()] = facilities["country"].apply(
            lambda x: 1 if x == country else 0
        )
        facility_to_merge_col.append("in_" + country.lower())

    for state in facilities.state.value_counts()[:10].keys():
        facilities["in_" + state.lower() + "_state"] = facilities["state"].apply(
            lambda x: 1 if x == state else 0
        )
        facility_to_merge_col.append("in_" + state.lower() + "_state")

    for city in facilities.city.value_counts()[:10].keys():
        facilities["in_" + city.lower() + "_city"] = facilities["city"].apply(
            lambda x: 1 if x == city else 0
        )
        facility_to_merge_col.append("in_" + city.lower() + "_city")

    train_df = train_df.merge(
        facilities[["facility_id"] + facility_to_merge_col], how="left"
    )
    feature_list = "num_trials_conducted, num_conditions_conducted, num_interventions_conducted, num_sponsors_conducted, num_phase1, num_phase2, num_phase3, num_phase4,avg_history_success_rate,  avg_history_adverse_events, max_historical_trial_enrollment, avg_historical_trial_enrollment,operation_years, years_since_last_trial_facility, in_multi_center_trial, num_multi_center_trial".split(
        ","
    )
    feature_list = [i.strip() for i in feature_list]

    expanded_df = pd.DataFrame(
        train_df["derived_features"].tolist(), columns=feature_list
    )
    train_df = train_df.drop(columns=["derived_features"])
    train_df = pd.concat([train_df, expanded_df], axis=1)

    return train_df


# From userstudy ipynb
def get_success_rate(transaction, column, nct_id, timestamp, outcome_analyses):
    nct_history = transaction[
        (
            transaction[column].isin(
                transaction[transaction.nct_id == nct_id][column].values
            )
        )
        & (transaction.date < timestamp)
    ].nct_id.unique()
    outcome_temp = outcome_analyses[outcome_analyses.nct_id.isin(nct_history)]
    outcome_temp = outcome_temp[
        (outcome_temp.p_value_modifier.isnull())
        | (outcome_temp.p_value_modifier != ">")
    ]
    outcome_temp = outcome_temp[
        (outcome_temp.p_value >= 0)
        & (outcome_temp.p_value <= 1)
        & (outcome_temp.outcome_type == "Primary")
    ]
    nct2p = outcome_temp.groupby("nct_id").p_value.min()
    if len(nct2p) == 0:
        return 0
    else:
        return sum(nct2p <= 0.05) / len(nct2p)


# From userstudy ipynb
def get_features_study_outcome(orig_train_df):
    train_df = orig_train_df.copy()
    dataset = get_dataset(name="rel-trial", download=True)
    task = get_task("rel-trial", "study-outcome", download=True)
    tables = dataset.get_db().table_dict

    interventions_studies = tables["interventions_studies"].df
    conditions_studies = tables["conditions_studies"].df
    sponsors_studies = tables["sponsors_studies"].df
    facilities_studies = tables["facilities_studies"].df
    studies = tables["studies"].df
    designs = tables["designs"].df
    eligibilities = tables["eligibilities"].df
    sponsors = tables["sponsors"].df

    outcomes = tables["outcomes"].df
    facilities = tables["facilities"].df
    outcome_analyses = tables["outcome_analyses"].df
    outcome_analyses = outcome_analyses.merge(
        outcomes[["id", "outcome_type"]], left_on="outcome_id", right_on="id"
    )

    eligibilities["minimum_age"] = eligibilities["minimum_age"].apply(
        lambda x: int(x.split("Years")[0]) if (x is not None) and ("Years" in str(x)) else -1
    )
    eligibilities["maximum_age"] = eligibilities["maximum_age"].apply(
        lambda x: int(x.split("Years")[0]) if (x is not None) and ("Years" in str(x)) else -1
    )

    train_studies = train_df.merge(studies)

    train_studies["is_observational"] = train_studies.study_type.apply(
        lambda x: 1 if x in ["Observational", "Observational [Patient Registry]"] else 0
    )
    train_studies["is_interventional"] = train_studies.study_type.apply(
        lambda x: 1 if x in ["Interventional"] else 0
    )
    train_studies["is_expanded_access"] = train_studies.study_type.apply(
        lambda x: 1 if x in ["Expanded Access"] else 0
    )

    train_studies["is_phase_1"] = train_studies.phase.apply(
        lambda x: 1 if (x is not None) and ("Phase 1" in str(x)) else 0
    )
    train_studies["is_phase_2"] = train_studies.phase.apply(
        lambda x: 1 if (x is not None) and ("Phase 2" in str(x)) else 0
    )
    train_studies["is_phase_3"] = train_studies.phase.apply(
        lambda x: 1 if (x is not None) and ("Phase 3" in str(x)) else 0
    )
    train_studies["is_phase_4"] = train_studies.phase.apply(
        lambda x: 1 if (x is not None) and ("Phase 4" in str(x)) else 0
    )

    train_studies = train_studies[
        [
            "nct_id",
            "is_observational",
            "is_interventional",
            "is_expanded_access",
            "is_phase_1",
            "is_phase_2",
            "is_phase_3",
            "is_phase_4",
            "enrollment",
            "number_of_arms",
            "number_of_groups",
            "has_dmc",
            "is_fda_regulated_drug",
            "is_fda_regulated_device",
            "is_unapproved_device",
            "is_ppsd",
            "is_us_export",
            "biospec_retention",
            "plan_to_share_ipd",
        ]
    ]

    train_df = train_df.merge(train_studies)

    train_design = train_df.merge(designs)
    train_design["is_randomized"] = train_design.allocation.apply(
        lambda x: 1 if x == "Randomized" else 0
    )
    train_design["is_parallel_assignment"] = train_design.intervention_model.apply(
        lambda x: 1 if x == "Parallel Assignment" else 0
    )
    train_design["is_single_group_assignment"] = train_design.intervention_model.apply(
        lambda x: 1 if x == "Single Group Assignment" else 0
    )
    train_design["is_crossover_assignment"] = train_design.intervention_model.apply(
        lambda x: 1 if x == "Crossover Assignment" else 0
    )
    train_design["is_factorial_assignment"] = train_design.intervention_model.apply(
        lambda x: 1 if x == "Factorial Assignment" else 0
    )
    train_design["is_sequential_assignment"] = train_design.intervention_model.apply(
        lambda x: 1 if x == "Sequential Assignment" else 0
    )

    train_design["is_single_masking"] = train_design.masking.apply(
        lambda x: 1 if x == "Single" else 0
    )
    train_design["is_double_masking"] = train_design.masking.apply(
        lambda x: 1 if x == "Double" else 0
    )
    train_design["is_triple_masking"] = train_design.masking.apply(
        lambda x: 1 if x == "Triple" else 0
    )
    train_design["is_quadruple_masking"] = train_design.masking.apply(
        lambda x: 1 if x == "Quadruple" else 0
    )
    train_design["is_no_masking"] = train_design.masking.apply(
        lambda x: 1 if x == "None (Open Label)" else 0
    )

    train_design = train_design[
        [
            "nct_id",
            "is_randomized",
            "is_parallel_assignment",
            "is_single_group_assignment",
            "is_crossover_assignment",
            "is_factorial_assignment",
            "is_sequential_assignment",
            "is_single_masking",
            "is_double_masking",
            "is_triple_masking",
            "is_quadruple_masking",
            "is_no_masking",
            "primary_purpose",
            "subject_masked",
            "caregiver_masked",
            "investigator_masked",
            "outcomes_assessor_masked",
        ]
    ]

    train_df = train_df.merge(train_design)

    train_eligibility = train_df.merge(eligibilities)
    train_eligibility["is_non_probability_sample"] = train_eligibility[
        "sampling_method"
    ].apply(lambda x: 1 if x == "Non-Probability Sample" else 0)
    train_eligibility["is_female_only"] = train_eligibility["gender"].apply(
        lambda x: 1 if x == "Female" else 0
    )
    train_eligibility["is_male_only"] = train_eligibility["gender"].apply(
        lambda x: 1 if x == "Male" else 0
    )
    train_eligibility["accept_healthy_volunteer"] = train_eligibility[
        "healthy_volunteers"
    ].apply(lambda x: 1 if x == "Accepts Healthy Volunteers" else 0)
    train_eligibility["is_min_age_ge_60"] = train_eligibility["minimum_age"].apply(
        lambda x: 1 if x >= 60 else 0
    )
    train_eligibility["is_max_age_le_20"] = train_eligibility["maximum_age"].apply(
        lambda x: 1 if x <= 20 else 0
    )
    train_eligibility = train_eligibility[
        [
            "nct_id",
            "is_non_probability_sample",
            "is_female_only",
            "is_male_only",
            "accept_healthy_volunteer",
            "is_min_age_ge_60",
        ]
    ]

    train_df = train_df.merge(train_eligibility, how="left")

    train_df["num_history_intervention"] = train_df.parallel_apply(
        lambda x: (
            interventions_studies[
                (
                    interventions_studies.intervention_id.isin(
                        interventions_studies[
                            interventions_studies.nct_id == x.nct_id
                        ].intervention_id.values
                    )
                )
                & (interventions_studies.date < x.timestamp)
            ]
            .groupby("intervention_id")
            .nct_id.agg(len)
            .mean()
        ),
        axis=1,
    )
    train_df["num_history_condition"] = train_df.parallel_apply(
        lambda x: (
            conditions_studies[
                (
                    conditions_studies.condition_id.isin(
                        conditions_studies[
                            conditions_studies.nct_id == x.nct_id
                        ].condition_id.values
                    )
                )
                & (conditions_studies.date < x.timestamp)
            ]
            .groupby("condition_id")
            .nct_id.agg(len)
            .mean()
        ),
        axis=1,
    )
    train_df["num_history_sponsor"] = train_df.parallel_apply(
        lambda x: (
            sponsors_studies[
                (
                    sponsors_studies.sponsor_id.isin(
                        sponsors_studies[
                            sponsors_studies.nct_id == x.nct_id
                        ].sponsor_id.values
                    )
                )
                & (sponsors_studies.date < x.timestamp)
            ]
            .groupby("sponsor_id")
            .nct_id.agg(len)
            .mean()
        ),
        axis=1,
    )
    train_df["num_history_facility"] = train_df.parallel_apply(
        lambda x: (
            facilities_studies[
                (
                    facilities_studies.facility_id.isin(
                        facilities_studies[
                            facilities_studies.nct_id == x.nct_id
                        ].facility_id.values
                    )
                )
                & (facilities_studies.date < x.timestamp)
            ]
            .groupby("facility_id")
            .nct_id.agg(len)
            .mean()
        ),
        axis=1,
    )
    train_df.fillna(0, inplace=True)

    # Calculate derived features for temporal_studies
    eligibilities = tables["eligibilities"].df
    studies = tables["studies"].df
    outcomes = tables["outcomes"].df
    facilities = tables["facilities"].df
    outcome_analyses = tables["outcome_analyses"].df
    outcome_analyses = outcome_analyses.merge(
        outcomes[["id", "outcome_type"]], left_on="outcome_id", right_on="id"
    )

    # Age range
    eligibilities["age_range"] = (
        eligibilities["maximum_age"] - eligibilities["minimum_age"]
    )

    # Number of words in eligibility criteria
    eligibilities["num_of_words_eligibility_criteria"] = eligibilities[
        "criteria"
    ].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

    # Number of words in trial title and description
    studies["num_of_words_trial_title"] = studies["brief_title"].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    studies["num_of_words_trial_description"] = studies["official_title"].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )

    # Number of outcomes
    outcomes_counts = outcomes.groupby("nct_id").size().reset_index(name="num_outcomes")

    # In US
    facilities["in_us"] = facilities["country"].apply(
        lambda x: 1 if x in ["US", "United States"] else 0
    )
    nct_id2in_us = (
        facilities.merge(facilities_studies, on="facility_id")
        .groupby("nct_id")
        .in_us.max()
        .reset_index()
    )

    train_df["history_success_rate_facility"] = train_df.parallel_apply(
        lambda x: get_success_rate(
            facilities_studies, "facility_id", x.nct_id, x.timestamp, outcome_analyses
        ),
        axis=1,
    )
    train_df["history_success_rate_sponsor"] = train_df.parallel_apply(
        lambda x: get_success_rate(
            sponsors_studies, "sponsor_id", x.nct_id, x.timestamp, outcome_analyses
        ),
        axis=1,
    )
    train_df["history_success_rate_condition"] = train_df.parallel_apply(
        lambda x: get_success_rate(
            conditions_studies, "condition_id", x.nct_id, x.timestamp, outcome_analyses
        ),
        axis=1,
    )
    train_df["history_success_rate_intervention"] = train_df.parallel_apply(
        lambda x: get_success_rate(
            interventions_studies,
            "intervention_id",
            x.nct_id,
            x.timestamp,
            outcome_analyses,
        ),
        axis=1,
    )

    # Merge derived features
    derived_features = train_df

    derived_features = derived_features.merge(
        eligibilities[["nct_id", "age_range", "num_of_words_eligibility_criteria"]],
        on="nct_id",
        how="left",
    )
    derived_features = derived_features.merge(
        studies[
            ["nct_id", "num_of_words_trial_title", "num_of_words_trial_description"]
        ],
        on="nct_id",
        how="left",
    )
    derived_features = derived_features.merge(outcomes_counts, on="nct_id", how="left")
    derived_features.fillna(0, inplace=True)

    # Check if facilities are in US
    derived_features = derived_features.merge(nct_id2in_us, how="left")
    return derived_features


TASK_PARAMS = {
    "rel-stack-user-engagement": {
        "dir": "stack/user-engagement",
        "target_col": "contribution",
        "table_prefix": "user_engagement",
        "identifier_cols": ["OwnerUserId", "timestamp"],
    },
    "rel-stack-user-badge": {
        "dir": "stack/user-badge",
        "target_col": "WillGetBadge",
        "table_prefix": "user_badge",
        "identifier_cols": ["UserId", "timestamp"],
    },
    "rel-stack-post-votes": {
        "dir": "stack/post-votes",
        "target_col": "popularity",
        "table_prefix": "post_votes",
        "identifier_cols": ["PostId", "timestamp"],
    },
    "rel-amazon-user-churn": {
        "dir": "amazon/user-churn",
        "target_col": "churn",
        "table_prefix": "user_churn",
        "identifier_cols": ["customer_id", "timestamp"],
    },
    "rel-amazon-user-ltv": {
        "dir": "amazon/user-ltv",
        "target_col": "ltv",
        "table_prefix": "user_ltv",
        "identifier_cols": ["customer_id", "timestamp"],
    },
    "rel-amazon-item-churn": {
        "dir": "amazon/item-churn",
        "target_col": "churn",
        "table_prefix": "item_churn",
        "identifier_cols": ["product_id", "timestamp"],
    },
    "rel-amazon-item-ltv": {
        "dir": "amazon/item-ltv",
        "target_col": "ltv",
        "table_prefix": "item_ltv",
        "identifier_cols": ["product_id", "timestamp"],
    },
    "rel-hm-item-sales": {
        "dir": "hm/item-sales",
        "target_col": "sales",
        "table_prefix": "item_sales",
        "identifier_cols": ["article_id", "timestamp"],
    },
    "rel-hm-user-churn": {
        "dir": "hm/user-churn",
        "target_col": "churn",
        "table_prefix": "user_churn",
        "identifier_cols": ["customer_id", "timestamp"],
    },
    "rel-f1-driver-position": {
        "dir": "f1/driver-position",
        "target_col": "position",
        "table_prefix": "driver_position",
        "identifier_cols": ["driverId", "date"],
    },
    "rel-f1-driver-dnf": {
        "dir": "f1/driver-dnf",
        "target_col": "did_not_finish",
        "table_prefix": "driver_dnf",
        "identifier_cols": ["driverId", "date"],
    },
    "rel-f1-driver-top3": {
        "dir": "f1/driver-top3",
        "target_col": "qualifying",
        "table_prefix": "driver_top3",
        "identifier_cols": ["driverId", "date"],
    },
    "rel-event-user-repeat": {
        "dir": "event/user-repeat",
        "target_col": "target",
        "table_prefix": "user_repeat",
        "identifier_cols": ["user", "timestamp"],
    },
    "rel-event-user-ignore": {
        "dir": "event/user-ignore",
        "target_col": "target",
        "table_prefix": "user_ignore",
        "identifier_cols": ["user", "timestamp"],
    },
    "rel-event-user-attendance": {
        "dir": "event/user-attendance",
        "target_col": "target",
        "table_prefix": "user_attendance",
        "identifier_cols": ["user", "timestamp"],
    },
}

if __name__ == "__main__":
    main()
