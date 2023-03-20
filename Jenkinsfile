class PodTemplateFiles {
  private Map files = [
    'MThreads GPU': 'ci/templates/musa.yaml',
  ]

  public String getPodTemplateFile(String platform) {
    String file = files.get(platform)
    return file
  }
}

pipeline {
  parameters {
    choice(name: 'HARDWARD_PLATFORM', choices: ['MThreads GPU'], description: 'Target hardware platform')
  }

  environment {
    TEMPLATE_FILE = "${new PodTemplateFiles().getPodTemplateFile(HARDWARD_PLATFORM)}"
  }

  agent none

  stages {
    stage('Lint') {
      parallel {
        stage('Python Lint') {
          agent {
            kubernetes {
              yamlFile "${env.TEMPLATE_FILE}"
              defaultContainer "main"
            }
          }
          steps {
            container('main') {
              sh '/opt/conda/condabin/conda run -n test_environment /bin/bash tools/lint/pylint.sh'
            }
          }
        }
        stage('C++ Lint') {
          agent {
            kubernetes {
              yamlFile "${env.TEMPLATE_FILE}"
              defaultContainer "main"
            }
          }
          steps {
            container('main') {
              sh '/opt/conda/condabin/conda run -n test_environment /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
            }
          }
        }
      }
    }
    stage('Build') {
      agent {
        kubernetes {
          yamlFile "${env.TEMPLATE_FILE}"
          defaultContainer "main"
        }
      }
      steps {
        container('main') {
          sh '/opt/conda/condabin/conda run -n test_environment /bin/bash build.sh'
        }
      }
    }
    stage('Unit Test') {
      agent {
        kubernetes {
          yamlFile "${env.TEMPLATE_FILE}"
          defaultContainer "main"
        }
      }
      steps {
        container('main') {
          sh '/opt/conda/condabin/conda run -n test_environment /bin/bash scripts/run_unittest.sh'
        }
      }
    }
  }
}
