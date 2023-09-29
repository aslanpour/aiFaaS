if [ $SCORE -ge $PYLINT_THRESHOLD ]; then
              echo "Pylint score is $SCORE, which is acceptable."
else
              echo "Pylint score is $SCORE, which is below the threshold. Failing the job."
              exit 1
fi
