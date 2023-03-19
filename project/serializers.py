from rest_framework import serializers

class FeatureSerializer(serializers.Serializer):
    features=serializers.ListField(child=serializers.FloatField())
    diagnosis=serializers.CharField()

    class Meta:
        fields=('features',)

    def validate_features(self, features):
        if len(features)!=10:
            raise serializers.ValidationError(f"Length of features should be 10, {len(features)} was supplied.")
        return features