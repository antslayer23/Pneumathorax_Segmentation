def get_resource(
        service_account_json,
        base_url,
        project_id,
        cloud_region,
        dataset_id,
        fhir_store_id,
        resource_type,
        resource_id):
    """Gets a FHIR resource."""
    url = '{}/projects/{}/locations/{}'.format(base_url,
                                               project_id, cloud_region)

    resource_path = '{}/datasets/{}/fhirStores/{}/fhir/{}/{}'.format(
        url, dataset_id, fhir_store_id, resource_type, resource_id)

    # Make an authenticated API request
    session = get_session(service_account_json)

    headers = {
        'Content-Type': 'application/fhir+json;charset=utf-8'
    }

    response = session.get(resource_path, headers=headers)
    response.raise_for_status()

    resource = response.json()

    print(json.dumps(resource, indent=2))

    return resource

def main():


if __name__ == "__main__":
    main()
