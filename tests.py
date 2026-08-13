"""
Runs tests for the task group specified via the RDL_GROUP environment variable.
"""

import os

import torch
import torch.multiprocessing as mp
from testing.testing_pipeline import test_task
from torch_geometric.seed import seed_everything

mp.set_sharing_strategy('file_system')

def main():
    seed_everything(42)
    if os.getenv("RDL_CONTAINER") != "1":
        torch._dynamo.config.capture_scalar_outputs = True
        torch.set_float32_matmul_precision('high')
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "cache/inductor"
    group = os.getenv("RDL_GROUP", "0")
    task_list_1 = [
        # Binary Classification
        ("rel-f1", "driver-dnf"), # unbalanced, very low unique, high eccentricity for low shortest paths !!FEAT
        ("rel-avito", "searchstream-click"), # VERY unbalanced, autocomplete, all unique, small overlap
        ("rel-avito", "user-visits"), # unbalanced, med overlap, low eccentricity/shortest paths
        ("rel-avito", "searchinfo-isuserloggedon"), # balanced, autocomplete, no overlap
        ("rel-event", "user-ignore"), # unbalanced, high overlap, very low eccentricity/shortest paths !!FEAT
        ("rel-stack", "user-engagement"), # very unbalanced, very low unique but high overlap, low eccentricity/shortest paths !!FEAT
        ("rel-trial", "study-outcome"), # balanced, all unique, high shortest path !!FEAT (no sql)
        ("rel-trial", "eligibilities-adult"), # very unbalanced, autocomplete, high eccentricity/shortest path
        ("rel-amazon", "item-churn"), # balanced, high overlap, low eccentricity/shortest paths !!FEAT
        ("rel-ratebeer", "user-churn"), # completely balanced, high eccentricitys
    ]

    task_list_2 = [
        # Regression
        ("rel-trial", "site-success"), # balanced, very high eccentricity !!FEAT (no sql)
        ("rel-event", "users-birthyear"), # decently unbalanced, very low eccentricity, year values, all unique
        ("rel-amazon", "item-ltv"), # extremely unbalanced, short average path !!FEAT
        ("rel-amazon", "review-rating"), # decently balanced, very low eccentricity, all unique
        ("rel-avito", "ad-ctr"), # decently unbalanced, medium eccentricity/shortest paths
        ("rel-f1", "driver-position"), # very balanced, high eccentricity but low shortest paths !!FEAT
        ("rel-stack", "post-votes"), # unbalanced, low shortest path, medium-scale numeric value !!FEAT
        ("rel-trial", "studies-enrollment"), # EXTREMELY unbalanced, medium eccentricity/shortest path, all unique
        ("rel-ratebeer", "beer_ratings-total_score"), # very balanced, high eccentricity, all unique
    ]

    task_list_3 = [
        # Multiclass Classification
        ("rel-stack", "badges-class"), # very unbalanced, very few classes, all unique, high eccentricity but low shortest path
        ("rel-salt", "sales-group"), # most balanced, VERY many classes
        ("rel-arxiv", "author-category"), # good balance, entity, many classes (some 0), very high eccentricity
        ("rel-salt", "item-incoterms"), # unbalanced, few classes, all unique, low eccentricity
        ("rel-salt", "sales-office"), # extremely unbalanced, med classes, high eccentricity
        ("rel-salt", "sales-payterms"), # med balance, many classes
    ]

    device = torch.device("cuda")

    if group == "0":
        task_list = task_list_1 + task_list_2 + task_list_3
    elif group == "1":
        task_list = task_list_1
    elif group == "2":
        task_list = task_list_2
    elif group == "3":
        task_list = task_list_3

    for (dataset, task) in task_list:
        test_task(dataset, task, device)


if __name__ == "__main__":
    main()
