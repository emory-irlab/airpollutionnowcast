.PHONY: clean lint requirements train_model predict_model train_test_split

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = airpollutionnowcast
PYTHON_INTERPRETER = python3
ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# VARIABLES                                                                       #
#################################################################################

#CONFIG_PATH = config/parameters.ini
TRAIN_DATA_PATH = data/processed/train.csv
VALID_DATA_PATH = data/processed/valid.csv
TEST_DATA_PATH = data/processed/test.csv

#################################################################################
# PROJECT CONSTRUCT                                                                 #
#################################################################################
## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment.yml --name $(PROJECT_NAME)
	conda activate $(PROJECT_NAME)
else
	@echo ">>> CONDA NEEDED TO CREATE ENVIRONMENT"
endif

## Test python environment is setup correctly
test_environment:
	conda activate $(PROJECT_NAME)
	$(PYTHON_INTERPRETER) test_environment.py

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

#################################################################################
# COMMANDS                                                                      #
#################################################################################

### extract search trends
#data/interim/search.csv:
#	$(PYTHON_INTERPRETER) airpolnowcast/data/extract_search_trend.py $(CONFIG_PATH) $@
#
### extract pol labels
#data/interim/pol.csv:
#	$(PYTHON_INTERPRETER) airpolnowcast/data/extract_pol_label.py $(CONFIG_PATH) $@
#
### extract physical measurements data
#data/interim/phys.csv:
#	$(PYTHON_INTERPRETER) airpolnowcast/data/extract_phys_meas.py $(CONFIG_PATH) $@
#
### process physical measurements features
#data/interim/process_phys.csv: data/interim/phys.csv
#	$(PYTHON_INTERPRETER) airpolnowcast/data/process_phys_feature.py $(CONFIG_PATH) $< $@
#
### merge all data
#data/interim/merged.csv: data/interim/pol.csv data/interim/search.csv data/interim/process_phys.csv
#	$(PYTHON_INTERPRETER) airpolnowcast/data/merge_data_files.py $(CONFIG_PATH) $^ $@
#
### train test split into files
#train_test_split: data/interim/merged.csv
#	$(PYTHON_INTERPRETER) airpolnowcast/data/train_test_split.py $(CONFIG_PATH) $< $(TRAIN_DATA_PATH) $(VALID_DATA_PATH) $(TEST_DATA_PATH)
#
### train model
#train_model:
#	$(PYTHON_INTERPRETER) airpolnowcast/evaluation/train_model.py $(CONFIG_PATH) $(TRAIN_DATA_PATH) $(VALID_DATA_PATH)
#
### predict and get report
#predict_model:
#	$(PYTHON_INTERPRETER) airpolnowcast/evaluation/predict_model.py $(CONFIG_PATH) $(TEST_DATA_PATH)
#
### predict fine-tuning results
#predict_fine_tuning:
#	$(PYTHON_INTERPRETER) airpolnowcast/evaluation/predict_fine_tuning.py $(CONFIG_PATH) $(TRAIN_DATA_PATH) $(VALID_DATA_PATH) $(TEST_DATA_PATH)


##########
# variable to run train_test_split on unit test data
##########
# unit_test_config
UNIT_CONFIG_PATH = config/unit_test.ini

UNIT_TRAIN_DATA_PATH = data/processed/unit_test/train.csv
UNIT_VALID_DATA_PATH = data/processed/unit_test/valid.csv
UNIT_TEST_DATA_PATH = data/processed/unit_test/test.csv

# process physical measurements features
data/external/unit_test/process_phys.csv:
	$(PYTHON_INTERPRETER) src/data/process_phys_feature.py $(UNIT_CONFIG_PATH) data/external/unit_test/phys.csv $@

# merge all data
data/interim/unit_test/merged.csv:
	$(PYTHON_INTERPRETER) src/data/merge_data_files.py $(UNIT_CONFIG_PATH) data/external/unit_test/pol.csv data/external/unit_test/search.csv data/external/unit_test/process_phys.csv $@

# train test split into files
unit_train_test_split: data/interim/unit_test/merged.csv
	$(PYTHON_INTERPRETER) src/data/train_test_split.py $(UNIT_CONFIG_PATH) $< $(UNIT_TRAIN_DATA_PATH) $(UNIT_VALID_DATA_PATH) $(UNIT_TEST_DATA_PATH)

# unit test on build features
unit_build_features:
	$(PYTHON_INTERPRETER) src/features/build_features.ut.py -v $(UNIT_CONFIG_PATH) $(UNIT_TRAIN_DATA_PATH)



#########
# variable to run train model on word vectors (without dict training)
#########
CONFIG_PATH = config/word_vector.ini

## extract search trends
data/interim/search.csv:
	$(PYTHON_INTERPRETER) src/data/extract_search_trend.py $(CONFIG_PATH) $@

## extract pol labels
data/interim/pol.csv:
	$(PYTHON_INTERPRETER) src/data/extract_pol_label.py $(CONFIG_PATH) $@

## extract physical measurements data
data/interim/phys.csv:
	$(PYTHON_INTERPRETER) src/data/extract_phys_meas.py $(CONFIG_PATH) $@

## process physical measurements features
data/interim/process_phys.csv: data/interim/phys.csv
	$(PYTHON_INTERPRETER) src/data/process_phys_feature.py $(CONFIG_PATH) $< $@

## merge all data
data/interim/merged.csv: data/interim/pol.csv data/interim/search.csv data/interim/process_phys.csv
	$(PYTHON_INTERPRETER) src/data/merge_data_files.py $(CONFIG_PATH) $^ $@

## train test split into files
train_test_split: data/interim/merged.csv
	$(PYTHON_INTERPRETER) src/data/train_test_split.py $(CONFIG_PATH) $< $(TRAIN_DATA_PATH) $(VALID_DATA_PATH) $(TEST_DATA_PATH)

## train model
train_model:
	$(PYTHON_INTERPRETER) src/evaluation/train_model.py $(CONFIG_PATH) $(TRAIN_DATA_PATH) $(VALID_DATA_PATH)

## predict and get report
predict_model:
	$(PYTHON_INTERPRETER) src/evaluation/predict_model.py $(CONFIG_PATH) $(TEST_DATA_PATH)


#########
# variable to run train model on word vectors (without dict training) and only on seed queries
#########
SEED_CONFIG_PATH = config/word_vector_seed_queries.ini

# all the operations makefile same above
## extract search trends
## train model
SEED_train_model:
	$(PYTHON_INTERPRETER) src/evaluation/train_model.py $(SEED_CONFIG_PATH) $(TRAIN_DATA_PATH) $(VALID_DATA_PATH)

## predict and get report
SEED_predict_model:
	$(PYTHON_INTERPRETER) src/evaluation/predict_model.py $(SEED_CONFIG_PATH) $(TEST_DATA_PATH)



#########
# variable to run train model on word vectors (without dict training) and only on seed queries and on NO2 prediction
#########

NO2_CONFIG_PATH = config/NO2/word_vector_NO2.ini
NO2_TRAIN_DATA_PATH = data/processed/NO2/train.csv
NO2_VALID_DATA_PATH = data/processed/NO2/valid.csv
NO2_TEST_DATA_PATH = data/processed/NO2/test.csv

#change operations
## extract pol labels
data/interim/NO2/pol.csv:
	$(PYTHON_INTERPRETER) src/data/extract_pol_label.py $(NO2_CONFIG_PATH) $@

## merge all data
data/interim/NO2/merged.csv: data/interim/NO2/pol.csv data/interim/search.csv data/interim/process_phys.csv
	$(PYTHON_INTERPRETER) src/data/merge_data_files.py $(NO2_CONFIG_PATH) $^ $@

## train test split into files
NO2_train_test_split: data/interim/NO2/merged.csv
	$(PYTHON_INTERPRETER) src/data/train_test_split.py $(NO2_CONFIG_PATH) $< $(NO2_TRAIN_DATA_PATH) $(NO2_VALID_DATA_PATH) $(NO2_TEST_DATA_PATH)

## train model
NO2_train_model:
	$(PYTHON_INTERPRETER) src/evaluation/train_model.py $(NO2_CONFIG_PATH) $(NO2_TRAIN_DATA_PATH) $(NO2_VALID_DATA_PATH)

## predict and get report
NO2_predict_model:
	$(PYTHON_INTERPRETER) src/evaluation/predict_model.py $(NO2_CONFIG_PATH) $(NO2_TEST_DATA_PATH)


#########
# variable to run train model on word vectors (without dict training) and only on seed queries and on PM25 prediction
#########

PM25_CONFIG_PATH = config/PM25/word_vector_PM25.ini
PM25_TRAIN_DATA_PATH = data/processed/PM25/train.csv
PM25_VALID_DATA_PATH = data/processed/PM25/valid.csv
PM25_TEST_DATA_PATH = data/processed/PM25/test.csv

#change operations
## extract pol labels
data/interim/PM25/pol.csv:
	$(PYTHON_INTERPRETER) src/data/extract_pol_label.py $(PM25_CONFIG_PATH) $@

## merge all data
data/interim/PM25/merged.csv: data/interim/PM25/pol.csv data/interim/search.csv data/interim/process_phys.csv
	$(PYTHON_INTERPRETER) src/data/merge_data_files.py $(PM25_CONFIG_PATH) $^ $@

## train test split into files
PM25_train_test_split: data/interim/PM25/merged.csv
	$(PYTHON_INTERPRETER) src/data/train_test_split.py $(PM25_CONFIG_PATH) $< $(PM25_TRAIN_DATA_PATH) $(PM25_VALID_DATA_PATH) $(PM25_TEST_DATA_PATH)

## train model
PM25_train_model:
	$(PYTHON_INTERPRETER) src/evaluation/train_model.py $(PM25_CONFIG_PATH) $(PM25_TRAIN_DATA_PATH) $(PM25_VALID_DATA_PATH)

## predict and get report
PM25_predict_model:
	$(PYTHON_INTERPRETER) src/evaluation/predict_model.py $(PM25_CONFIG_PATH) $(PM25_TEST_DATA_PATH)
























## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f data/raw/*.csv
	rm -f data/interim/*.csv
	rm -f data/processed/*.csv
	rm -f models/*.h5
	rm -f models/*.pkl
	rm -f models/interim/*.h5
	rm -f reports/*.csv
	rm -f reports/*.ini



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
