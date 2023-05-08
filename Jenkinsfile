class PodTemplateFiles {
  private Map files = [
    'MThreads GPU': 'ci/templates/musa.yaml',
  ]

  public String getPodTemplateFile(String platform) {
    String file = files.get(platform)
    return file
  }
}

def ifTriggeredByTimer() {
  return currentBuild.getBuildCauses()[0].shortDescription == 'Started by timer'
}

pipeline {
  parameters {
    choice(name: 'HARDWARE_PLATFORM', choices: ['MThreads GPU'], description: 'Target hardware platform')
  }

  agent {
    kubernetes {
      yamlFile "${new PodTemplateFiles().getPodTemplateFile(params.HARDWARE_PLATFORM)}"
      defaultContainer "main"
    }
  }

  triggers {
    // UTC
    cron(env.BRANCH_NAME == 'main' ? '0 18 * * *' : '')
  }

  stages {
    stage('Lint') {
      parallel {
        stage('Python Lint') {
          steps {
            container('main') {
              sh 'git config --global --add safe.directory \"*\"'
              sh '/opt/conda/condabin/conda run -n test_environment --no-capture-output /bin/bash tools/lint/pylint.sh'
            }
          }
        }
        stage('C++ Lint') {
          steps {
            container('main') {
              sh 'git config --global --add safe.directory \"*\"'
              sh '/opt/conda/condabin/conda run -n test_environment --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
            }
          }
        }
      }
    }
    stage('Build') {
      steps {
        container('main') {
          sh '/bin/bash --login scripts/update_daily_mudnn.sh'
          sh '/bin/bash --login build.sh'
        }
      }
    }
    stage('Unit Test') {
      steps {
        container('main') {
          sh '/bin/bash --login scripts/run_unittest.sh'
        }
      }
    }
    stage('Integration Test') {
      steps {
        container('main') {
          sh '/bin/bash --login scripts/run_integration_test.sh'
        }
      }
    }
    stage('Daily Release') {
      agent {
        kubernetes {
          yamlFile 'ci/templates/musa.yaml'
          defaultContainer "main"
        }
      }
      when {
        allOf {
          branch 'main'
          expression { ifTriggeredByTimer() }
        }
      }
      steps {
        container('main') {
          sh '/bin/bash --login scripts/update_daily_mudnn.sh'
          // Build wheel packages under python3.8, using the existing conda environment
          sh '/bin/bash --login -c "/opt/conda/condabin/conda run -n test_environment --no-capture-output /bin/bash scripts/build_wheel.sh"'
          // Copy built wheel packages to shared directory "/artifacts"
          sh 'cp dist/*.whl /artifacts/ && cp ${PYTORCH_REPO_PATH}/dist/*.whl /artifacts/'

          // Build wheel packages under python3.9, create a new conda environment
          sh '/bin/bash --login -c "/opt/conda/condabin/conda env create -f docker/common/conda-env-torch_musa-py39.yaml" && \
              /opt/conda/condabin/conda run -n py39 --no-capture-output pip install -r docker/common/requirements-py39.txt -i \
              https://pypi.tuna.tsinghua.edu.cn/simple some-package'
          sh '/bin/bash --login -c "/opt/conda/condabin/conda run -n py39 --no-capture-output /bin/bash scripts/build_wheel.sh"'
          sh 'cp dist/*.whl /artifacts/ && cp ${PYTORCH_REPO_PATH}/dist/*.whl /artifacts/'
          
          // Add some description
          sh 'echo "commit id: "$(git rev-parse HEAD) > /artifacts/README.txt'
          sh 'echo "dependencies: " >> /artifacts/README.txt && \
              DAILY_MUDNN_REL_DIR=$(find ./ -name "daily_mudnn*" | awk -F/ \'NR==1{print $NF}\') && \
              echo "daily_mudnn:"$(find ${DAILY_MUDNN_REL_DIR} -name "*.txt" | awk -F/ \'{print $NF}\' | awk -F_ \'{print $1}\') >> /artifacts/README.txt && \
              cat .musa_dependencies >> /artifacts/README.txt'
        }
        container('release') {
          // Publish new release to oss (minio)
          sh 'oss-release /artifacts/'
        }
      }
    }
  }
}
