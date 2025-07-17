def Pipeline(String DockerImg, String DockerRunArgs, String GpuType) {
    docker.image("${DockerImg}").inside("${DockerRunArgs}") {
        gitlabCommitStatus(name: "01-${GpuType}-lint check", state: "running") {
            // add safe directory
            sh 'git config --global --add safe.directory \"*\"'
            // lint check
            sh '/opt/conda/condabin/conda run -n py310 --no-capture-output /bin/bash tools/lint/pylint.sh'
            sh '/opt/conda/condabin/conda run -n py310 --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
        }
        gitlabCommitStatus(name: "02-${GpuType}-env prepare", state: "running") {
            // update musa sdk
            sh 'pip uninstall torch torch_musa -y'
            // sh '/bin/bash --login docker/common/release/update_release_all.sh'
            sh 'pip install -r requirements_ci.txt -i https://pypi.tuna.tsinghua.edu.cn/simple'
            sh 'wget -O /tmp/flamegraph.pl https://oss.mthreads.com/mt-ai-data/integration-test/flamegraph.pl && chmod 755 /tmp/flamegraph.pl'
        }
        gitlabCommitStatus(name: "03-${GpuType}-build torch", state: "running") {
            // build
            sh '/bin/bash --login -c "KINETO_URL=https://sh-code.mthreads.com/ai/kineto.git conda run -n py310 --no-capture-output /bin/bash build.sh -c"'
        }
        parallel (
            UNITTEST: {
                gitlabCommitStatus(name: "04-${GpuType}-basic unit tests", state: "running") {
                    // unit test
                    sh "export MUSA_VISIBLE_DEVICES=0,1"
                    sh "GPU_TYPE=${GpuType} FLAMEGRAPH_PL_SCRIPT=/tmp/flamegraph.pl /bin/bash --login scripts/run_unittest.sh"
                }
            },
            FSDP: {
                gitlabCommitStatus(name: "04-${GpuType}-fsdp unit tests", state: "running") {
                    // unit test
                    sh "export MUSA_VISIBLE_DEVICES=2,3"
                    sh "GPU_TYPE=${GpuType} /bin/bash --login scripts/run_fsdp.sh"
                }
            },
            INTEGRATION: {
                gitlabCommitStatus(name: "05-${GpuType}-integration tests", state: "running") {
                    // integration test
                    sh "export MUSA_VISIBLE_DEVICES=4,5"
                    sh "MUSA_VISIBLE_DEVICES=4,5 GPU_TYPE=${GpuType} /bin/bash --login scripts/run_integration_test.sh"
                }
            }
        )
    }
}

pipeline {
  agent none

  options {
    gitLabConnection('sh-code')
  }

  environment {
    S4000IMG = 'sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py310:rc4.0.0-v2.0.0-pt25-qy2'
    S5000IMG = 'sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py310:rc4.0.0-v2.0.0-pt25-ph1'
    DOCKER_RUN_ARGS = '--network=host ' +
      '--user root ' +
      '--privileged ' +
      '--shm-size 20G ' +
      '--pid=host ' +
      '-e TARGET_DEVICE=musa ' +
      '-e PYTORCH_REPO_PATH=/home/pytorch ' +
      '-e MTHREADS_VISIBLE_DEVICES=all ' +
      '-e MUSA_VISIBLE_DEVICES=all ' +
      '-v /home/mccxadmin/torch_musa_integration/data:/data/torch_musa_integration/local ' +
      '-v /juicefs/torch_musa_integration/data:/data/torch_musa_integration/shared'
  }

  stages {
    stage('Run task in parallel') {
      failFast true
      parallel {
        stage('S4000') {
          agent {
            label 'te-s4000'
          }
          steps {
            timeout(time: 200, unit: 'MINUTES') {
              script {
                Pipeline("${S4000IMG}", "${DOCKER_RUN_ARGS}", "S4000")
              }
            }
          }
        }
        stage('S5000') {
          agent {
            label 's5000'
          }
          steps {
            timeout(time: 200, unit: 'MINUTES') {
              script {
                Pipeline("${S5000IMG}", "${DOCKER_RUN_ARGS}", "S5000")
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
