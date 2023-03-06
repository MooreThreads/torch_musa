pipeline {
  agent none
  stages {
    stage('Build') {
      agent {
        kubernetes {
          yamlFile "${env.TEMPLATE_FILE}"
          defaultContainer 'main'
        }

      }
      steps {
        container(name: 'main') {
          sh '/bin/bash --login -c "conda run -n py39 /bin/bash tools/lint/pylint.sh"'
        }

      }
    }

    stage('Unit Test') {
      agent {
        kubernetes {
          yamlFile "${env.TEMPLATE_FILE}"
          defaultContainer 'main'
        }

      }
      steps {
        container(name: 'main') {
          sh '/bin/bash --login -c "conda run -n py39 /bin/bash scripts/run_unittest.sh"'
        }

      }
    }

  }
  environment {
    TEMPLATE_FILE = "${new PodTemplateFiles().getPodTemplateFile(HARDWARD_PLATFORM)}"
  }
  parameters {
    choice(name: 'HARDWARD_PLATFORM', choices: ['MThreads GPU'], description: 'Target hardware platform')
  }
}