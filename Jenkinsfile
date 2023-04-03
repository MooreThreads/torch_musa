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

  agent {
    kubernetes {
      yamlFile "${new PodTemplateFiles().getPodTemplateFile(HARDWARD_PLATFORM)}"
      defaultContainer "main"
    }
  }

  stages {
    stage('Lint') {
      parallel {
        stage('Python Lint') {
          steps {
            container('main') {
              sh '/opt/conda/condabin/conda run -n test_environment --no-capture-output /bin/bash tools/lint/pylint.sh'
            }
          }
        }
        stage('C++ Lint') {
          steps {
            container('main') {
              sh '/opt/conda/condabin/conda run -n test_environment --no-capture-output /bin/bash tools/lint/git-clang-format.sh --rev origin/main'
            }
          }
        }
      }
    }
    stage('Build') {
      steps {
        container('main') {
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
  }
}
