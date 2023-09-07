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

  environment {
    RUN_NEXT_STAGE = true    // Whether to run the next stage in daily CI stages
    DAILY_UT_PASSED = false  // Whether daily unit test passed
    DAILY_IT_PASSED = false  // Whether daily integration test passed
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
              sh '/opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/pylint.sh'
            }
          }
        }
        stage('C++ Lint') {
          steps {
            container('main') {
              sh 'git config --global --add safe.directory \"*\"'
              sh '/opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
            }
          }
        }
      }
    }
    stage('Parallel Build & Test') {
      parallel {
        stage('Stable Build & Test') {
          stages {
            stage('Build') {
              steps {
                sh 'git config --global --add safe.directory \"*\"'
                sh '/bin/bash --login scripts/update_release_mudnn.sh'
                sh '/bin/bash --login -c "conda run -n py38 --no-capture-output /bin/bash build.sh"'
              }
            }
            stage('Unit Test') {
              steps {
                sh '/bin/bash --login scripts/run_unittest.sh'
              }
            }
            stage('Integration Test') {
              steps {
                sh '/bin/bash --login scripts/run_integration_test.sh'
              }
            }
          }
        }
        stage('Daily Build & Test') {
          agent {
            kubernetes {
              yamlFile 'ci/templates/musa.yaml'
              defaultContainer "main"
            }
          }
          when {
            beforeAgent true
            expression { !ifTriggeredByTimer() }
          }
          stages {
            stage('Build') {
              steps {
                container('main') {
                  script {
                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                      sh 'git config --global --add safe.directory \"*\"'
                      sh '/bin/bash --login scripts/update_daily_musart.sh'
                      sh '/bin/bash --login scripts/update_daily_mudnn.sh'
                      sh '/bin/bash --login -c "conda run -n py38 --no-capture-output /bin/bash build.sh"'
                    }
                  }
                }
              }
              post {
                success {
                  echo 'BUILD SUCCESS!'
                }
                failure {
                  script {
                    echo 'BUILD FAILURE!'
                    env.RUN_NEXT_STAGE = false
                  }
                }
              }
            }
            stage('Unit Test') {
              when {
                expression { env.RUN_NEXT_STAGE }
              }
              steps {
                container('main') {
                  script {
                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                      sh '/bin/bash --login scripts/run_unittest.sh'
                    }
                  }
                }
              }
              post {
                success {
                  script {
                    echo 'Unit Test SUCCESS!'
                    env.DAILY_UT_PASSED = true
                  }
                }
                failure {
                  script {
                    echo 'Unit Test FAILURE!'
                  }
                }
              }
            }
            stage('Integration Test') {
              when {
                expression { env.RUN_NEXT_STAGE }
              }
              steps {
                container('main') {
                  script {
                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                      sh '/bin/bash --login scripts/run_integration_test.sh'
                    }
                  }
                }
              }
              post {
                success {
                  script {
                    echo 'Integration Test SUCCESS!'
                    env.DAILY_IT_PASSED = true
                  }
                }
                failure {
                  echo 'Integration Test FAILURE!'
                }
              }
            }
          }
        }
      }
    }
    stage('Daily Release') {
      when {
        // beforeAgent true
        allOf {
          branch 'main'
          expression { ifTriggeredByTimer() }
        }
      }
      steps {
        container('main') {
          sh '/bin/bash --login -c "BUILD_ARTIFACTS=1 /bin/bash scripts/run_daily_release.sh"'
        }
        container('release') {
          // Publish new release to oss (minio)
          sh '/bin/bash --login -c "PUBLISH_ARTIFACTS=1 /bin/bash scripts/run_daily_release.sh"'
        }
      }
    }
  }
}
