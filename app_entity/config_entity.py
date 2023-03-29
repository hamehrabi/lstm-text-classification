# configuration entity
from collections import namedtuple

DatasetConfig = namedtuple("DatasetConfig", ["name", "schema", "buffer_size", "batch_size"])
PreprocessingConfig = namedtuple("PreprocessingConfig", ["vocal_size"])

ModelTrainingConfig = namedtuple("TrainingConfig", ["model_save_dir",
                                                    "model_checkpoint_dir",
                                                    "model_root_dir",
                                                    "epoch",
                                                    "tensorboard_log_dir",
                                                    "base_accuracy",
                                                    "validation_step",

                                                    ])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir",
                                                               "training_pipeline_obj_dir",
                                                               "training_pipeline_obj_file_name",
                                                               "training_pipeline_obj_file_path",
                                                               "execution_report_dir",
                                                               "execution_report_file_name",
                                                               "execution_report_file_path"
                                                               ])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["change_threshold", ])

ModelDeploymentConfig = namedtuple("ModelDeploymentConfig", ["model_serving_dir"])
