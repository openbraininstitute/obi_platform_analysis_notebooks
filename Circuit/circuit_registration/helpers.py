"""Helper functions for circuit registration notebook."""

from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from entitysdk import Client, models
from entitysdk.types import (
    CircuitBuildCategory,
    CircuitScale,
    DerivationType,
    TargetSimulator,
)


def query_metadata_options(
    client: Client, default_species: str = "Rattus norvegicus"
) -> dict:
    """Query all metadata options from entitycore for circuit registration.

    Args:
        client: entitysdk Client instance.
        default_species: Default species name to pre-select.

    Returns:
        Dict with all queried metadata options.
    """
    print("Collecting metadata options...")

    # Query available species
    print("> Querying species")
    all_species = client.search_entity(entity_type=models.Species).all()
    species_names = sorted([s.name for s in all_species])
    if default_species not in species_names:
        default_species = species_names[0]

    # Query available circuits (exclude single-neuron scale)
    print("> Querying circuits")
    circuit_scales = [scale.name for scale in CircuitScale if scale != "single"]
    all_circuits = client.search_entity(
        entity_type=models.Circuit, query={"scale__in": circuit_scales}
    ).all()
    circuit_names = [
        (f"{c.scale}: {c.name}{' [public]' if c.authorized_public else ''}", c.name)
        for c in all_circuits
    ]
    circuit_names = sorted(circuit_names, key=lambda c: c[0])

    # Query available subjects
    print("> Querying subjects")
    all_subjects = client.search_entity(entity_type=models.Subject).all()
    subject_species = set(s.species for s in all_subjects)
    subject_names = {
        species.name: sorted(
            [
                s.name
                for s in all_subjects
                if s.name != "Unknown" and s.species == species
            ]
        )
        for species in subject_species
    }

    # Query available brain region hierarchies
    print("> Querying brain region hierarchies")
    all_hierarchies = client.search_entity(
        entity_type=models.BrainRegionHierarchy
    ).all()
    hierarchy_species = set(h.species for h in all_hierarchies)
    hierarchy_names = {
        species.name: sorted([h.name for h in all_hierarchies if h.species == species])
        for species in hierarchy_species
    }

    # Query available brain regions (grouped by hierarchy)
    print("> Querying brain regions")
    all_brain_regions = client.search_entity(entity_type=models.BrainRegion).all()
    brain_region_names = {
        h.name: sorted([r.name for r in all_brain_regions if r.hierarchy_id == h.id])
        for h in all_hierarchies
    }

    # Query available licenses
    print("> Querying licenses")
    all_licenses = client.search_entity(entity_type=models.License).all()
    license_names = sorted([lic.label for lic in all_licenses])

    # Query available contributors
    print("> Querying contributors")
    all_persons = client.search_entity(entity_type=models.Person).all()
    person_names = sorted(set([p.pref_label for p in all_persons]))

    all_organizations = client.search_entity(entity_type=models.Organization).all()
    organization_names = sorted(set([o.pref_label for o in all_organizations]))

    all_consortia = client.search_entity(entity_type=models.Consortium).all()
    consortium_names = sorted(set([c.pref_label for c in all_consortia]))

    # Query available publications
    print("> Querying publications")
    all_publications = client.search_entity(entity_type=models.Publication).all()
    publication_options = sorted(
        [
            (
                f"{p.DOI} - {p.authors[0]['family_name']}{' et al.' if len(p.authors) > 1 else ''}, {p.publication_year}: {p.title}",
                p.DOI,
            )
            for p in all_publications
        ],
        key=lambda x: x[0],
    )

    print("Done.")

    return {
        "species_names": species_names,
        "default_species": default_species,
        "circuit_names": circuit_names,
        "subject_names": subject_names,
        "hierarchy_names": hierarchy_names,
        "brain_region_names": brain_region_names,
        "license_names": license_names,
        "person_names": person_names,
        "organization_names": organization_names,
        "consortium_names": consortium_names,
        "publications": publication_options,
    }


def create_metadata_widgets(options: dict) -> dict:
    """Create interactive widgets for circuit metadata entry.

    Args:
        options: Dict returned by query_metadata_options().

    Returns:
        Dict of widget instances keyed by field name.
    """
    species_names = options["species_names"]
    default_species = options["default_species"]
    circuit_names = options["circuit_names"]
    subject_names = options["subject_names"]
    hierarchy_names = options["hierarchy_names"]
    brain_region_names = options["brain_region_names"]
    license_names = options["license_names"]

    # --- Required fields ---
    w_name = widgets.Text(
        value="",
        placeholder="e.g. EXAMPLE_01__N_10__top_nodes_dim6",
        description="Name:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    w_description = widgets.Textarea(
        value="",
        placeholder="e.g. Example circuit with 10 neurons",
        description="Description:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="60px"),
    )

    w_build_category = widgets.Dropdown(
        options=[(e.name, e) for e in CircuitBuildCategory],
        description="Build category:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
    )

    w_species = widgets.Dropdown(
        options=species_names,
        value=default_species,
        description="Species:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
    )

    w_subject = widgets.Dropdown(
        options=subject_names.get(default_species, []),
        description="Subject:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
    )

    w_brain_region_hierarchy = widgets.Dropdown(
        options=hierarchy_names.get(default_species, []),
        description="Brain region hierarchy:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    w_brain_region = widgets.Dropdown(
        options=brain_region_names.get(w_brain_region_hierarchy.value, []),
        description="Brain region:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    # --- Species-dependent observers ---
    def _update_subjects(change):
        w_subject.options = subject_names.get(change["new"], [])

    def _update_hierarchies(change):
        w_brain_region_hierarchy.options = hierarchy_names.get(change["new"], [])

    def _update_brain_regions(change):
        w_brain_region.options = brain_region_names.get(change["new"], [])

    w_species.observe(_update_subjects, names="value")
    w_species.observe(_update_hierarchies, names="value")
    w_brain_region_hierarchy.observe(_update_brain_regions, names="value")

    # --- Optional fields ---
    w_target_simulator = widgets.Dropdown(
        options=[("None (use default)", None)] + [(e.name, e) for e in TargetSimulator],
        value=None,
        description="Target simulator:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
    )

    w_root = widgets.Dropdown(
        options=[("None", None)] + circuit_names,
        value=None,
        description="Root circuit:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    w_parent = widgets.Dropdown(
        options=[("None", None)] + circuit_names,
        value=None,
        description="Parent circuit:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    w_derivation_type = widgets.Dropdown(
        options=[
            (e.name, e)
            for e in DerivationType
            if e.name in ("circuit_extraction", "circuit_rewiring")
        ],
        description="Derivation type:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
        disabled=True,
    )

    def _toggle_derivation(change):
        w_derivation_type.disabled = change["new"] is None

    w_parent.observe(_toggle_derivation, names="value")

    w_license = widgets.Dropdown(
        options=[("None", None)] + [(n, n) for n in license_names],
        value=None,
        description="License:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
    )

    w_published_in = widgets.Text(
        value="",
        placeholder="e.g. Reimann et al and Isbister et al",
        description="Published in:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    w_contact = widgets.Text(
        value="",
        placeholder="e.g. support@openbraininstitute.org",
        description="Contact e-mail:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%"),
    )

    w_experiment_date = widgets.DatePicker(
        description="Experiment date:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="50%"),
    )

    # --- Layout ---
    required_box = widgets.VBox(
        [
            widgets.HTML("<h4>Required</h4>"),
            w_name,
            w_description,
            w_build_category,
            w_species,
            w_subject,
            w_brain_region_hierarchy,
            w_brain_region,
        ]
    )

    optional_box = widgets.VBox(
        [
            widgets.HTML("<h4>Optional</h4>"),
            w_target_simulator,
            w_root,
            w_parent,
            w_derivation_type,
            w_license,
            w_published_in,
            w_contact,
            w_experiment_date,
        ]
    )

    display(required_box, optional_box)

    return {
        "w_name": w_name,
        "w_description": w_description,
        "w_build_category": w_build_category,
        "w_species": w_species,
        "w_subject": w_subject,
        "w_brain_region_hierarchy": w_brain_region_hierarchy,
        "w_brain_region": w_brain_region,
        "w_target_simulator": w_target_simulator,
        "w_root": w_root,
        "w_parent": w_parent,
        "w_derivation_type": w_derivation_type,
        "w_license": w_license,
        "w_published_in": w_published_in,
        "w_contact": w_contact,
        "w_experiment_date": w_experiment_date,
    }


def assemble_circuit_metadata(w: dict, options: dict) -> dict:
    """Assemble and validate circuit_metadata dict from widget values.

    Args:
        w: Dict of widget instances returned by create_metadata_widgets().
        options: Dict returned by query_metadata_options().

    Returns:
        Validated circuit_metadata dict.

    Raises:
        ValueError: If required fields are missing or circuit name already exists.
    """
    circuit_metadata = {
        "name": w["w_name"].value,
        "description": w["w_description"].value,
        "build_category": w["w_build_category"].value,
        "species": w["w_species"].value,
        "subject": w["w_subject"].value,
        "brain_region_hierarchy": w["w_brain_region_hierarchy"].value,
        "brain_region": w["w_brain_region"].value,
        "target_simulator": w["w_target_simulator"].value,
        "root": w["w_root"].value or None,
        "parent": w["w_parent"].value or None,
        "derivation_type": w["w_derivation_type"].value
        if not w["w_derivation_type"].disabled
        else None,
        "license": w["w_license"].value or None,
        "published_in": w["w_published_in"].value or None,
        "contact": w["w_contact"].value or None,
        "experiment_date": w["w_experiment_date"].value.strftime("%d.%m.%Y")
        if w["w_experiment_date"].value
        else None,
    }

    # Validate required fields
    _required = [
        "name",
        "description",
        "build_category",
        "species",
        "subject",
        "brain_region_hierarchy",
        "brain_region",
    ]
    _missing = [k for k in _required if not circuit_metadata.get(k)]
    if circuit_metadata.get("parent") and not circuit_metadata.get("derivation_type"):
        _missing.append("derivation_type")
    if _missing:
        raise ValueError(f"Missing required metadata fields: {_missing}")

    # Validate circuit name uniqueness
    existing_names = [name for _, name in options["circuit_names"]]
    if circuit_metadata["name"] in existing_names:
        raise ValueError(f"Circuit name '{circuit_metadata['name']}' already exists!")

    print("circuit_metadata:")
    for k, v in circuit_metadata.items():
        print(f"  {k}: {v!r}")

    return circuit_metadata


def validate_paths(
    circuit_path: str,
    overview_image_path: str | None,
    sim_designer_image_path: str | None,
) -> None:
    """Check that provided file paths exist and print errors for missing ones.

    Args:
        circuit_path: Path to the compressed SONATA circuit.
        overview_image_path: Optional path to overview image.
        sim_designer_image_path: Optional path to simulation designer image.
    """
    missing = []
    for name, path in [
        ("Circuit path", circuit_path),
        ("Overview image path", overview_image_path),
        ("Sim designer image path", sim_designer_image_path),
    ]:
        if path and not Path(path).exists():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError("File(s) not found:\n  " + "\n  ".join(missing))

    print("All file paths OK.")


def create_contributor_widgets(options: dict) -> dict:
    """Create SelectMultiple widgets for contributor selection.

    Args:
        options: Dict returned by query_metadata_options().

    Returns:
        Dict with widget instances: "w_persons", "w_organizations", "w_consortia".
    """
    w_persons = widgets.SelectMultiple(
        options=options["person_names"],
        description="Persons:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="150px"),
    )

    w_organizations = widgets.SelectMultiple(
        options=options["organization_names"],
        description="Organizations:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="150px"),
    )

    w_consortia = widgets.SelectMultiple(
        options=options["consortium_names"],
        description="Consortia:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="150px"),
    )

    display(w_persons, w_organizations, w_consortia)

    return {
        "w_persons": w_persons,
        "w_organizations": w_organizations,
        "w_consortia": w_consortia,
    }


def assemble_contributions(contrib_widgets: dict) -> dict:
    """Assemble circuit_contributions dict from contributor widget selections.

    Args:
        contrib_widgets: Dict returned by create_contributor_widgets().

    Returns:
        Dict mapping contributor names to their type and role.
    """
    circuit_contributions = {}
    for name in contrib_widgets["w_persons"].value:
        circuit_contributions[name] = {"type": "person", "role": "unspecified"}
    for name in contrib_widgets["w_organizations"].value:
        circuit_contributions[name] = {"type": "organization", "role": "unspecified"}
    for name in contrib_widgets["w_consortia"].value:
        circuit_contributions[name] = {"type": "consortium", "role": "unspecified"}

    print(f"{len(circuit_contributions)} contributor(s) selected:")
    for name, info in circuit_contributions.items():
        print(f"  {name} ({info['type']})")

    return circuit_contributions


def create_publication_widgets(options: dict) -> dict:
    """Create SelectMultiple widgets for publication selection.

    Args:
        options: Dict returned by query_metadata_options().

    Returns:
        Dict with widget instances: "w_pub_source", "w_pub_component", "w_pub_application".
    """
    publication_options = options["publications"]

    w_pub_source = widgets.SelectMultiple(
        options=publication_options,
        description="Source:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="150px"),
    )

    w_pub_component = widgets.SelectMultiple(
        options=publication_options,
        description="Component:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="150px"),
    )

    w_pub_application = widgets.SelectMultiple(
        options=publication_options,
        description="Application:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="80%", height="150px"),
    )

    display(w_pub_source, w_pub_component, w_pub_application)

    return {
        "w_pub_source": w_pub_source,
        "w_pub_component": w_pub_component,
        "w_pub_application": w_pub_application,
    }


def assemble_publications(pub_widgets: dict) -> dict:
    """Assemble circuit_publications dict from publication widget selections.

    Args:
        pub_widgets: Dict returned by create_publication_widgets().

    Returns:
        Dict mapping DOIs to their publication type.

    Raises:
        ValueError: If the same DOI is selected in multiple categories.
    """
    circuit_publications = {}
    duplicates = []

    for doi in pub_widgets["w_pub_source"].value:
        circuit_publications[doi] = {"type": "entity_source"}
    for doi in pub_widgets["w_pub_component"].value:
        if doi in circuit_publications:
            duplicates.append(doi)
        else:
            circuit_publications[doi] = {"type": "component_source"}
    for doi in pub_widgets["w_pub_application"].value:
        if doi in circuit_publications:
            duplicates.append(doi)
        else:
            circuit_publications[doi] = {"type": "application"}

    if duplicates:
        raise ValueError(
            f"Duplicate DOI(s) selected in multiple categories: {set(duplicates)}"
        )

    print(f"{len(circuit_publications)} publication(s) selected:")
    for doi, info in circuit_publications.items():
        print(f"  {doi} ({info['type']})")

    return circuit_publications


def check_registered_circuit(client: "Client", registered_circuit) -> None:
    """Fetch and print info about a registered circuit entity.

    Args:
        client: entitysdk Client instance.
        registered_circuit: The registered circuit object (or None if not registered).
    """
    if registered_circuit:
        # Fetch entity incl. assets
        res = client.get_entity(
            entity_id=registered_circuit.id, entity_type=models.Circuit
        )

        # Print circuit info
        print(f"Circuit '{res.name}' registered (ID {res.id})")
        print(
            f"  scale={res.scale}, neurons={res.number_neurons}, synapses={res.number_synapses}, connections={res.number_connections}"
        )
        print(
            f"  has_morphologies={res.has_morphologies}, has_point_neurons={res.has_point_neurons}, has_electrical_cell_models={res.has_electrical_cell_models}, has_spines={res.has_spines}"
        )

        # Print asset info
        print(f"Assets registered ({len(res.assets)}):")
        for a in res.assets:
            print(
                f"  Asset '{a.label}' ({'dir' if a.is_directory else 'file'}) registered (ID {a.id})"
            )
    else:
        print("Circuit not registered!")


def create_dry_run_widget():
    """Create a radio button widget for selecting dry run or actual registration.

    Returns:
        Widget instance with value True (dry run) or False (actual registration).
    """
    w_dry_run = widgets.RadioButtons(
        options=[("Dry run", True), ("Actual registration", False)],
        value=True,
        description="Mode:",
        style={"description_width": "180px"},
    )
    display(w_dry_run)
    return w_dry_run
