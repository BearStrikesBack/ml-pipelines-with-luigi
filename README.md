# About

Some cool stuff with ML orchestrated by Docker and Luigi, presented by Pweave.

# How this project is organized
* `download_data`. Download data.
* `process_data`. Process data. Generate features. Make Train/Test split.
* `train_models`. Train models. Train linear regression and lightgbm on Train dataset.
* `evaluate_models`. Evaluate models. Calculate metric perfomance on Test dataset for both models. Plot some chats.
* `make_report`. Make report. Present results of the whole pipeline.

# How to run
* Build docker images

    `bash build-task-images.sh 0.1`

* Run pipeline, write logs to output file

    `docker-compose up orchestrator |& tee ./output.log`

* Clean containers

    `bash docker-clean.sh`

# Ways to improve
1. Create base docker image with most of the libraries and add layers to it instead of building each time from `python:3.6-slim`. Currently takes about 90 sec to build images on clean system from scratch.
2. Use more sophisticated ML algorithms; Use more feature engineering; Use parameter tuning.