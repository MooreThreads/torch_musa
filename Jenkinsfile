class PodTemplateFiles {

  private Map files = [
    'MThreads GPU S4000': 'ci/templates/musa_s4000.yaml',
    'MThreads GPU S3000': 'ci/templates/musa_s3000.yaml',
    // 'MThreads GPU S80': 'ci/templates/musa_s80.yaml',
  ]

  public String getPodTemplateFile(String platform) {
    String file = files.get(platform)
    return file
  }

}

def ifTriggeredByTimer() {
  return currentBuild.getBuildCauses()[0].shortDescription == 'Started by timer'
}

@NonCPS
def cancelPreviousBuilds() {
  def jobName = env.JOB_NAME
  def buildNumber = env.BUILD_NUMBER.toInteger()

    /* the following APIs like getItemByFullName, getListener etc must be added to ScriptApproval in Jenkins */
  def currentJob = Jenkins.instance.getItemByFullName(jobName)
  def gitlabMergeRequestIid = env.gitlabMergeRequestIid

    /* Iterating over the builds for specific job */
  for (def build : currentJob.builds) {
    def listener = build.getListener()
    def exec = build.getExecutor()
        /* If there is a build that is currently running and it's not current build */
    if (build.isBuilding() && build.number.toInteger() < buildNumber && exec != null && build.envVars.gitlabMergeRequestIid == gitlabMergeRequestIid) {
            /* Then stop it */
      exec.interrupt(
                    Result.ABORTED,
                    new CauseOfInterruption.UserInterruption("Aborted by #${buildNumber}")
                )
      println("Aborted previously running build #${build.number}")
    }
  }
}

pipeline {
  agent {
    kubernetes {
      yamlFile "${new PodTemplateFiles().getPodTemplateFile('MThreads GPU S4000')}"
      defaultContainer 'main'
    }
  }

  post {
      unstable {
            script {
                currentBuild.result = 'FAILURE'
                error("Build marked as FAILURE due to instability.")
            }
            updateGitlabCommitStatus name: 'build', state: 'failed'
        }
      failure {
        updateGitlabCommitStatus name: 'build', state: 'failed'
      }
      success {
        updateGitlabCommitStatus name: 'build', state: 'success'
      }
      aborted {
        updateGitlabCommitStatus name: 'build', state: 'canceled'
      }
  }

  options {
    gitLabConnection('gitlab')
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
    stage('Prepare') {
      steps {
        updateGitlabCommitStatus name: 'build', state: 'pending'
        container('main') {
          withCredentials([string(credentialsId: 'mt-ai-ops-gitlab-api-key', variable: 'GITLAB_API_KEY')]) {
            sh 'git config --global url."https://mt-ai-ops:${GITLAB_API_KEY}@sh-code.mthreads.com".insteadOf "https://sh-code.mthreads.com"'
          }
        }
        script {
          cancelPreviousBuilds()
        }
      }
    }

    stage('Lint') {
      parallel {
        stage('Python Lint') {
          steps {
            container('main') {
              sh 'git config --global --add safe.directory \"*\"'
              sh '/opt/conda/condabin/conda run -n py38 --no-capture-output pip uninstall torch_musa -y && /bin/bash tools/lint/pylint.sh'
            }
          }
        }
        stage('C++ Lint') {
          steps {
            container('main') {
              sh 'git config --global --add safe.directory \"*\"'
              sh 'CHECK_BRANCH_NAME=${CHANGE_TARGET:-"main"} && /opt/conda/condabin/conda run -n py38 --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/${CHECK_BRANCH_NAME}'
            }
          }
        }
      }
    }

    stage('Start building') {
      steps {
        updateGitlabCommitStatus name: 'build', state: 'running'
      }
    }

    stage('Parallel Build & Test') {
      failFast true
      parallel {
        stage('S4000 Stable Build & Test') {
          agent {
            kubernetes {
              yamlFile "${new PodTemplateFiles().getPodTemplateFile('MThreads GPU S4000')}"
              defaultContainer 'main'
            }
          }
          stages {
            stage('Build') {
              steps {
                withCredentials([string(credentialsId: 'mt-ai-ops-gitlab-api-key', variable: 'GITLAB_API_KEY')]) {
                  sh 'git config --global url."https://mt-ai-ops:${GITLAB_API_KEY}@sh-code.mthreads.com".insteadOf "https://sh-code.mthreads.com"'
                }
                sh 'git config --global --add safe.directory \"*\"'
                sh 'git config --unset-all core.hooksPath'
                sh 'pip uninstall torch_musa -y'
                sh '/bin/bash --login docker/common/release/update_release_all.sh'
                sh '/bin/bash --login -c "KINETO_URL=https://sh-code.mthreads.com/ai/kineto.git conda run -n py38 --no-capture-output /bin/bash build.sh -c"'
              }
            }
            stage('Unit Test') {
              steps {
                script {
                  lock('s4000-ut-lock') {
                    sh '/bin/bash --login -c "conda run -n py38 --no-capture-output python scripts/run_unittest_dist.py --gpu-type S4000"'
                  }
                }
              }
            }
            stage('Integration Test') {
              steps {
                sh 'GPU_TYPE=S4000 /bin/bash --login scripts/run_integration_test.sh'
              }
            }
            stage('Archive Test Reports and Artifacts') {
              steps {
                sh '''
                    cwd=$(pwd) find build/reports -type f -name '*.xml' -execdir bash -c 'for file; do python ${cwd}/scripts/extract_failed_tests.py ${file}; done' bash {} +
                '''
                junit allowEmptyResults: true, testResults: 'build/reports/**/*.xml'
                sh '''
                    find dist -type f -name '*.egg' -execdir bash -c 'for file; do mv "$file" "${file%/*}/S4000_${file##*/}"; done' bash {} +
                '''
                archiveArtifacts artifacts: 'dist/*.egg', fingerprint: true, allowEmptyArchive: true
              }
            }
          }
        }
        stage('s3000 Stable Build & Test') {
          agent {
            kubernetes {
              yamlFile "${new PodTemplateFiles().getPodTemplateFile('MThreads GPU S3000')}"
              defaultContainer 'main'
            }
          }
          stages {
            stage('Build') {
              steps {
                container('main') {
                  withCredentials([string(credentialsId: 'mt-ai-ops-gitlab-api-key', variable: 'GITLAB_API_KEY')]) {
                    sh 'git config --global url."https://mt-ai-ops:${GITLAB_API_KEY}@sh-code.mthreads.com".insteadOf "https://sh-code.mthreads.com"'
                  }
                }
                sh 'git config --global --add safe.directory \"*\"'
                sh 'git config --unset-all core.hooksPath'
                sh 'pip uninstall torch_musa -y'
                sh '/bin/bash --login docker/common/release/update_release_all.sh'
                sh '/bin/bash --login -c "KINETO_URL=https://sh-code.mthreads.com/ai/kineto.git conda run -n py38 --no-capture-output /bin/bash build.sh -c"'
              }
            }
            stage('Unit Test') {
              steps {
                script {
                  lock('s3000-ut-lock') {
                    sh '/bin/bash --login -c "conda run -n py38 --no-capture-output python scripts/run_unittest_dist.py --gpu-type S3000"'
                  }
                }
              }
            }
            stage('Integration Test') {
              steps {
                sh 'GPU_TYPE=S3000 /bin/bash --login scripts/run_integration_test.sh'
              }
            }
            stage('Archive Test Reports and Artifacts') {
              steps {
                sh '''
                    cwd=$(pwd) find build/reports -type f -name '*.xml' -execdir bash -c 'for file; do python ${cwd}/scripts/extract_failed_tests.py ${file}; done' bash {} +
                '''
                junit allowEmptyResults: true, testResults: 'build/reports/**/*.xml'
                sh '''
                    find dist -type f -name '*.egg' -execdir bash -c 'for file; do mv "$file" "${file%/*}/S3000_${file##*/}"; done' bash {} +
                '''
                archiveArtifacts artifacts: 'dist/*.egg', fingerprint: true, allowEmptyArchive: true
              }
            }
          }
        }
      }
    }
  }
}
