# OCTO_TOKEN=`cat /egr/research-dselab/renjie3/renjie/octoai.txt`
# echo $OCTO_TOKEN

curl --request GET \
  --url https://api.octoai.cloud/v1/logs/tune_01hvsx4fd0e1grgz1w2fys4qnc/stream \
  --header 'Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiIwMzFiZjNmNC01NzZmLTQ3NGItOTc0NC0xM2RkNGU2YmViMTgiLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiJjNzMzMzdlNS0yYzk5LTRkZDMtYjJjMC02Mzg4ZmJjZjQ2ZmUiLCJ1c2VySWQiOiI3OWUyOTJlZC1mYmQ5LTRjN2UtYTIxMi0zMGJkNDRiMGJjMmMiLCJyb2xlcyI6WyJGRVRDSC1ST0xFUy1CWS1BUEkiXSwicGVybWlzc2lvbnMiOlsiRkVUQ0gtUEVSTUlTU0lPTlMtQlktQVBJIl0sImF1ZCI6IjNkMjMzOTQ5LWEyZmItNGFiMC1iN2VjLTQ2ZjYyNTVjNTEwZSIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkub2N0b21sLmFpIiwiaWF0IjoxNzEzMzk2ODUzfQ.RWVXDYFd_Kxg-avR2vytOyEEuVUoi0OP8MACtAoDbN4Ac2M_SyjQdye_ujakkvdLWCyylghCw4joXw85sm4a_ZYx56BhGvpjzKhj526qBaSBJ3wQjL0xIpO0nKPxvSsnhEdIsnZeloCBcKvhYSCE5zTd6JH35xXH0mbH6jNV0LvoP3DmayHYg2EwyDuxabmnsxGrO2CDTS97IUxkgIEGbDrHqCYE30FcO8HtW3bacFRCcjuEa3QQa8cGLFC172DhVX77ZTePtRhmpmh7e2FZusR7GpzTPPe0TJ--08ay3Yqzai7n370KwwyU7AqQLOKNkBS1oYvNlml7NlCmbNjGkA'