# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dictionary mapping each subject to its respective subcategories.
Subcategories represent more specific topics which are later grouped into broader categories.
"""

subcategories = {
    # STEM Subjects
    "abstract_algebra": ["math"],
    "college_mathematics": ["math"],
    "elementary_mathematics": ["math"],
    "high_school_mathematics": ["math"],
    "high_school_statistics": ["math"],
    "college_chemistry": ["chemistry"],
    "high_school_chemistry": ["chemistry"],
    "college_physics": ["physics"],
    "astronomy": ["physics"],
    "conceptual_physics": ["physics"],
    "high_school_physics": ["physics"],
    "electrical_engineering": ["engineering"],
    "college_biology": ["biology"],
    "high_school_biology": ["biology"],
    "college_computer_science": ["computer science"],
    "high_school_computer_science": ["computer science"],
    "computer_security": ["computer science"],
    "machine_learning": ["computer science"],
    # Humanities
    "formal_logic": ["philosophy"],
    "logical_fallacies": ["philosophy"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "philosophy": ["philosophy"],
    "world_religions": ["philosophy"],
    "high_school_european_history": ["history"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "prehistory": ["history"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "professional_law": ["law"],
    # Social Sciences
    "econometrics": ["economics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_microeconomics": ["economics"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "us_foreign_policy": ["politics"],
    "high_school_government_and_politics": ["politics"],
    "high_school_psychology": ["psychology"],
    "professional_psychology": ["psychology"],
    "sociology": ["culture"],
    "human_sexuality": ["culture"],
    "high_school_geography": ["geography"],
    # Other (Business, Health, Miscellaneous)
    "business_ethics": ["business"],
    "management": ["business"],
    "marketing": ["business"],
    "professional_accounting": ["other"],
    "global_facts": ["other"],
    "miscellaneous": ["other"],
    "anatomy": ["health"],
    "clinical_knowledge": ["health"],
    "college_medicine": ["health"],
    "human_aging": ["health"],
    "medical_genetics": ["health"],
    "nutrition": ["health"],
    "professional_medicine": ["health"],
    "virology": ["health"],
}

# Dictionary mapping broader categories to their corresponding subcategories.
# Categories help in grouping subcategories into broader disciplines for analysis.

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}
