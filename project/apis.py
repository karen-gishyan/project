from rest_framework.mixins import CreateModelMixin
from rest_framework.viewsets import GenericViewSet
from project.serializers import FeatureSerializer
from rest_framework.response import Response
import environ
import boto3
import json



#NOTE swagger does not display mixins with GenericAPIView
class UploadFeatures(GenericViewSet,CreateModelMixin):
    serializer_class=FeatureSerializer
    env=environ.Env()


    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        client=boto3.client('lambda',region_name=self.env('AWS_REGION_NAME'),
                            aws_access_key_id=self.env('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=self.env('AWS_SECRET_ACCESS_KEY'))

        features=str(serializer.data['features'])
        diagnosis=serializer.data['diagnosis']
        response=client.invoke(FunctionName='search',Payload=json.dumps({'features':features,'diagnosis':diagnosis}))
        data=json.loads(response['Payload'].read())
        try:
            # if no body return the error message
            data['visualization_url']=f"http://127.0.0.1:8000/display-graph/{data['body']['image_url']}"
        except:
            data=data['errorMessage']
        return Response(data)
