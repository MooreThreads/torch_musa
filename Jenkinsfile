pipeline {
  agent none

  options {
    gitLabConnection('sh-code')
  }

  environment {
    S3000IMG = 'sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py38:rc3.1.0-v1.3.0-qy1'
    S4000IMG = 'sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py38:rc3.1.0-v1.3.0-qy2'
    DOCKER_RUN_ARGS = '--network=host ' +
      '--user root ' +
      '--privileged ' +
      '--shm-size 20G ' +
      '-e TARGET_DEVICE=musa ' +
      '-e PYTORCH_REPO_PATH=/home/pytorch ' +
      '-e MTHREADS_VISIBLE_DEVICES=all ' +
      '-e MUSA_VISIBLE_DEVICES=all ' +
      '-v /home/mccxadmin/torch_musa_integration/data:/data/torch_musa_integration/local ' +
      '-v /juicefs/torch_musa_integration/data:/data/torch_musa_integration/shared'
  }

  stages {
    stage('Run task in parallel') {
      parallel {
        stage('S3000') {
          agent {
            label 's3000'
          }
          steps {
            script {
              docker.image("${S3000IMG}").inside("${DOCKER_RUN_ARGS}") {
                gitlabCommitStatus(name: '01-s3000-lint check', state: 'running') {
                  // add safe directory
                  sh 'git config --global --add safe.directory \"*\"'
                  // lint check
                  sh '/opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/pylint.sh'
                  sh '/opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
                }
                gitlabCommitStatus(name: '02-s3000-env prepare', state: 'running') {
                  // update musa sdk
                  sh 'pip uninstall torch torch_musa -y'
                  sh '/bin/bash --login docker/common/release/update_release_all.sh'
                }
                gitlabCommitStatus(name: '03-s3000-build torch', state: 'running') {
                  // build
                  sh '/bin/bash --login -c "KINETO_URL=https://sh-code.mthreads.com/ai/kineto.git conda run -n py38 --no-capture-output /bin/bash build.sh -c"'
                }
                gitlabCommitStatus(name: '04-s3000-unit tests', state: 'running') {
                  // unit test
                  sh 'GPU_TYPE=S3000 /bin/bash --login scripts/run_unittest.sh'
                }
                gitlabCommitStatus(name: '05-s3000-integration tests', state: 'running') {
                  // integration test
                  sh 'pip install transformers datasets'
                  sh 'GPU_TYPE=S3000 /bin/bash --login scripts/run_integration_test.sh'
                }
              }
            }
          }
        }
        stage('S4000') {
          agent {
            label 's4000'
          }
          steps {
            script {
              docker.image("${S4000IMG}").inside("${DOCKER_RUN_ARGS}") {
                gitlabCommitStatus(name: '01-s4000-lint check', state: 'running') {
                  // add safe directory
                  sh 'git config --global --add safe.directory \"*\"'
                  // lint check
                  sh '/opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/pylint.sh'
                  sh '/opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
                }
                gitlabCommitStatus(name: '02-s4000-env prepare', state: 'running') {
                  // update musa sdk
                  sh 'pip uninstall torch torch_musa -y'
                  sh '/bin/bash --login docker/common/release/update_release_all.sh'
                }
                gitlabCommitStatus(name: '03-s4000-build torch', state: 'running') {
                  // build
                  sh '/bin/bash --login -c "KINETO_URL=https://sh-code.mthreads.com/ai/kineto.git conda run -n py38 --no-capture-output /bin/bash build.sh -c"'
                }
                gitlabCommitStatus(name: '04-s4000-unit tests', state: 'running') {
                  // unit test
                  sh 'GPU_TYPE=S4000 /bin/bash --login scripts/run_unittest.sh'
                }
                gitlabCommitStatus(name: '05-s4000-integration tests', state: 'running') {
                  // integration test
                  sh 'pip install transformers datasets'
                  sh 'GPU_TYPE=S4000 /bin/bash --login scripts/run_integration_test.sh'
                }
              }
            }
          }
        }
      }
    }
  }

  post {
    unstable {
      script {
        currentBuild.result = 'FAILURE'
        error("Build marked as FAILURE due to instability.")
      }
      updateGitlabCommitStatus name: '06-final', state: 'failed'
    }
    failure {
      updateGitlabCommitStatus name: '06-final', state: 'failed'
    }
    success {
      updateGitlabCommitStatus name: '06-final', state: 'success'
    }
    aborted {
      updateGitlabCommitStatus name: '06-final', state: 'canceled'
    }
  }
}
