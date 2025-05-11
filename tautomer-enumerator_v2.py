import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, rdDepictor
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import tempfile
import os
from PIL import Image
import shutil
from streamlit.runtime.caching import cache_data

st.set_page_config(page_title="Tautomer Enumerator", layout="wide")


def enumerate_tautomers(smiles, max_tautomers=100):
    """
    Core function to enumerate tautomers using RDKit's tautomer enumerator.
    """
    st.write(f"Processing SMILES: {smiles}")
    
    # First, check if the rdMolStandardize module is available
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        has_rdmolstandardize = True
    except ImportError:
        st.error("Your RDKit installation does not have rdMolStandardize module available.")
        return None, "Missing rdMolStandardize module"
    
    # Parse input molecule with failsafes
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            # Try alternate sanitization approach
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception as e:
        st.error(f"Error parsing SMILES: {e}")
        return None, f"Error parsing SMILES: {e}"
    
    if mol is None:
        return None, "Invalid SMILES string or molecule cannot be parsed"
    
    # Create tautomer enumerator with custom settings
    enumerator = rdMolStandardize.TautomerEnumerator()
    
    # Configure the enumerator for best results
    enumerator.SetMaxTautomers(max_tautomers)  # Set upper limit
    
    # Ensure we have hydrogens for proper enumeration
    mol_with_h = Chem.AddHs(mol)
    
    # Perform enumeration
    try:
        tautomers = enumerator.Enumerate(mol_with_h)
        st.write(f"RDKit enumeration returned object of type: {type(tautomers)}")
    except Exception as e:
        st.error(f"Tautomer enumeration failed: {e}")
        return [mol_with_h], mol_with_h
    
    # Extract tautomers into a list
    taut_list = []
    
    try:
        # Handle different RDKit versions and return types
        if hasattr(tautomers, 'GetNumTautomers'):
            # Newer RDKit API
            st.write(f"Enumeration found {tautomers.GetNumTautomers()} tautomers")
            for i in range(tautomers.GetNumTautomers()):
                taut = tautomers.GetTautomer(i)
                taut_list.append(taut)
                #st.write(f"Added tautomer {i+1}: {Chem.MolToSmiles(taut)}")
        elif hasattr(tautomers, 'count') and callable(getattr(tautomers, 'count')):
            # Alternative API
            st.write(f"Enumeration found {tautomers.count()} tautomers")
            for i in range(tautomers.count()):
                taut = tautomers.get(i)
                taut_list.append(taut)
                #st.write(f"Added tautomer {i+1}: {Chem.MolToSmiles(taut)}")
        elif isinstance(tautomers, (list, tuple)):
            # Direct list return
            st.write(f"Enumeration returned a list with {len(tautomers)} tautomers")
            taut_list.extend(list(tautomers))
            for i, taut in enumerate(taut_list):
                st.write(f"Added tautomer {i+1}: {Chem.MolToSmiles(taut)}")
        elif 'TautomerEnumeratorResult' in str(type(tautomers)):
            # Handle TautomerEnumeratorResult type
            try:
                # Try to access as an iterable
                st.write("Handling TautomerEnumeratorResult type")
                taut_count = 0
                
                # First try by iteration
                try:
                    for taut in tautomers:
                        taut_list.append(taut)
                        taut_count += 1
                        #st.write(f"Added tautomer {taut_count}: {Chem.MolToSmiles(taut)}")
                except:
                    st.write("Could not iterate directly over TautomerEnumeratorResult")
                
                # If that fails, try using indices
                if taut_count == 0:
                    try:
                        # Try using len() and indices
                        taut_count = len(tautomers)
                        st.write(f"Result has {taut_count} tautomers")
                        
                        for i in range(taut_count):
                            taut = tautomers[i]
                            taut_list.append(taut)
                            #st.write(f"Added tautomer {i+1}: {Chem.MolToSmiles(taut)}")
                    except:
                        st.write("Could not access TautomerEnumeratorResult by indices")
                
                # If both methods fail, try GetTautomers method
                if taut_count == 0:
                    try:
                        if hasattr(tautomers, 'GetTautomers'):
                            st.write("Using GetTautomers method")
                            all_tautomers = tautomers.GetTautomers()
                            for i, taut in enumerate(all_tautomers):
                                taut_list.append(taut)
                                #st.write(f"Added tautomer {i+1}: {Chem.MolToSmiles(taut)}")
                    except Exception as e:
                        st.write(f"GetTautomers method failed: {e}")
                
                # If no tautomers were added, fallback to adding original molecule
                if len(taut_list) == 0:
                    st.write("No tautomers could be extracted, using original molecule")
                    taut_list.append(mol_with_h)
            except Exception as e:
                st.warning(f"Error processing TautomerEnumeratorResult: {e}")
                taut_list.append(mol_with_h)
        else:
            # Fallback for unknown return type
            st.warning(f"Unknown tautomer enumeration return type: {type(tautomers)}")
            # Add the input molecule as a fallback
            taut_list.append(mol_with_h)
            
            # Try specific approach for your RDKit version
            try:
                st.write("Attempting direct method calls on unknown object")
                
                # Try methods we know exist in some versions
                if hasattr(tautomers, 'size'):
                    count = tautomers.size()
                    st.write(f"Found 'size' method: {count} tautomers")
                    for i in range(count):
                        taut_list.append(tautomers[i])
                elif hasattr(tautomers, '__len__'):
                    count = len(tautomers)
                    st.write(f"Found '__len__' method: {count} tautomers")
                    for i in range(count):
                        taut_list.append(tautomers[i])
                
                # Last resort: try direct attribute access for debugging
                st.write("Available attributes/methods:")
                for attr in dir(tautomers):
                    if not attr.startswith('__'):
                        st.write(f"- {attr}")
            except Exception as e:
                st.write(f"Additional debugging failed: {e}")
    except Exception as e:
        st.error(f"Error processing tautomers: {e}")
        # Fallback to original molecule
        taut_list.append(mol_with_h)
    
    # Always ensure we have at least one molecule
    if len(taut_list) == 0:
        st.warning("No tautomers were generated, using original molecule")
        taut_list.append(mol_with_h)
    
    # Get the canonical tautomer
    try:
        canonical_taut = enumerator.Canonicalize(mol_with_h)
        st.write(f"Canonical tautomer: {Chem.MolToSmiles(canonical_taut)}")
    except Exception as e:
        st.warning(f"Error getting canonical tautomer: {e}")
        canonical_taut = mol_with_h  # Fallback to original
    
    # Logging final results
    st.write(f"Final tautomer count: {len(taut_list)}")
    
    return taut_list, canonical_taut


def display_molecule_svg(mol, legend="", width=300, height=250):
    """Display a molecule as SVG"""
    if mol is None:
        return f"<p>{legend} (No molecule to display)</p>"
    
    try:
        drawer = Draw.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol, legend=legend)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace('xmlns:svg', 'xmlns')
        return svg
    except Exception as e:
        return f"<p>Error displaying molecule: {str(e)}</p>"
        
def add_hydrogens_carefully(mol):
    """
    Add hydrogens incrementally to a molecule, stopping when adding more would
    cause energy calculation failures. This avoids over-adding hydrogens that
    create invalid valence states.
    """
    if mol is None:
        return None
        
    try:
        # First strip all hydrogens
        mol_no_h = Chem.RemoveAllHs(mol, sanitize=False)
        
        # Sanitize the heavy atom structure
        try:
            Chem.SanitizeMol(mol_no_h)
        except:
            # If we can't sanitize even the heavy atoms, try conversion via SMILES
            try:
                smiles = Chem.MolToSmiles(mol_no_h)
                mol_no_h = Chem.MolFromSmiles(smiles)
                if mol_no_h is None:
                    return None
            except:
                return None
        
        # First try: standard hydrogen addition with MMFF check
        try:
            mol_with_h = Chem.AddHs(mol_no_h)
            
            # Check if we can calculate energy with this structure
            props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
            if props is not None:
                # We have a valid structure
                return mol_with_h
        except:
            # Standard hydrogen addition didn't work, proceed to incremental method
            pass
        
        # If standard method failed, use explicit incremental hydrogen addition
        st.write("Using incremental hydrogen addition...")
        
        # Create a copy to work with
        result_mol = Chem.Mol(mol_no_h)
        
        # Find atoms that typically need hydrogens
        h_addition_sites = []
        for atom in result_mol.GetAtoms():
            symbol = atom.GetSymbol()
            valence = atom.GetExplicitValence()
            
            # Simple rules for common atoms:
            # C: Add H to get to valence 4
            if symbol == 'C' and valence < 4:
                h_addition_sites.append((atom.GetIdx(), 4 - valence))
            # N: Add H to get to valence 3 (unless already has double bond)
            elif symbol == 'N' and valence <= 3:
                # Check if N has a double bond
                has_double_bond = False
                for bond in atom.GetBonds():
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        has_double_bond = True
                        break
                
                if has_double_bond:
                    # If N has a double bond, add at most 1 H
                    if valence < 3:
                        h_addition_sites.append((atom.GetIdx(), 1))
                else:
                    # Otherwise add H to get to valence 3
                    h_addition_sites.append((atom.GetIdx(), 3 - valence))
            # O: Add H to get to valence 2 (unless already has double bond)
            elif symbol == 'O' and valence < 2:
                h_addition_sites.append((atom.GetIdx(), 2 - valence))
        
        # Get MMFF properties of the base molecule
        can_calculate_energy = False
        try:
            # Try to embed and calculate energy for the base molecule
            mol_3d = Chem.Mol(result_mol)
            AllChem.EmbedMolecule(mol_3d)
            props = AllChem.MMFFGetMoleculeProperties(mol_3d)
            if props is not None:
                can_calculate_energy = True
        except:
            can_calculate_energy = False
        
        # If base molecule allows energy calculation, we're good
        if can_calculate_energy:
            return result_mol
            
        # Otherwise, try adding hydrogens one by one to each site
        editable_mol = Chem.EditableMol(result_mol)
        
        for atom_idx, num_h_to_add in h_addition_sites:
            # Add hydrogens one at a time
            for i in range(num_h_to_add):
                # Add a hydrogen to the current atom
                h_idx = editable_mol.AddAtom(Chem.Atom('H'))
                editable_mol.AddBond(atom_idx, h_idx, Chem.BondType.SINGLE)
                
                # Convert to regular molecule to test
                test_mol = editable_mol.GetMol()
                
                # Check if we can calculate energy with this structure
                try:
                    mol_3d = Chem.Mol(test_mol)
                    AllChem.EmbedMolecule(mol_3d)
                    props = AllChem.MMFFGetMoleculeProperties(mol_3d)
                    if props is None:
                        # This hydrogen made the molecule invalid, remove it
                        editable_mol.RemoveAtom(h_idx)
                        break
                except:
                    # Something went wrong, remove this hydrogen
                    editable_mol.RemoveAtom(h_idx)
                    break
        
        # Return the final molecule with carefully added hydrogens
        return editable_mol.GetMol()
    
    except Exception as e:
        st.warning(f"Error in incremental hydrogen addition: {e}")
        # Fallback to the original molecule
        return mol


def calculate_mmff_energy_reliable(mol):
    """
    Calculate MMFF energy using RDKit, ensuring reliable results.
    Returns None if energy cannot be calculated.
    """
    if mol is None:
        return None
    
    try:
        # Create a copy for 3D
        mol_3d = Chem.Mol(mol)
        
        # Generate a 3D conformation
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        
        if result == -1:
            return None
        
        # Set up MMFF force field
        props = AllChem.MMFFGetMoleculeProperties(mol_3d)
        if props is None:
            return None
            
        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props)
        if ff is None:
            return None
        
        # Calculate energy
        energy = ff.CalcEnergy()
        return energy
    except Exception as e:
        return None

def generate_3d_for_mol_reliable(mol):
    """
    Generate a 3D conformation for a molecule.
    Uses the molecule as-is, assuming it's already been fixed.
    """
    if mol is None:
        return None
    
    try:
        # Create a copy for 3D
        mol_3d = Chem.Mol(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        
        if result == -1:
            # Fall back to 2D
            AllChem.Compute2DCoords(mol_3d)
            return mol_3d
        
        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
        except:
            try:
                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
            except:
                pass
                
        return mol_3d
    except Exception as e:
        # Fall back to the input molecule
        return mol

def simple_sanitize_tautomer(mol):
    """
    Simple sanitization that prioritizes displaying the structure
    even if energy calculation isn't possible.
    """
    if mol is None:
        return None
        
    try:
        # Remove all hydrogens first
        mol_no_h = Chem.RemoveAllHs(mol, sanitize=False)
        
        # Try to sanitize heavy atom structure
        try:
            mol_sanitized = Chem.Mol(mol_no_h)
            Chem.SanitizeMol(mol_sanitized, 
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except:
            # If sanitization fails, try via SMILES
            try:
                smiles = Chem.MolToSmiles(mol_no_h)
                mol_sanitized = Chem.MolFromSmiles(smiles)
                if mol_sanitized is None:
                    return mol  # Return original if all else fails
            except:
                return mol  # Return original if all else fails
        
        # Add back hydrogens
        try:
            mol_with_h = Chem.AddHs(mol_sanitized)
            return mol_with_h
        except:
            return mol_sanitized  # Return without H if adding H fails
    except Exception as e:
        st.warning(f"Simple sanitization failed: {e}")
        return mol  # Return original if all fails

def cleanup_tautomer_smiles(smiles):
    """
    Apply chemical cleanup rules to make tautomer SMILES more reasonable.
    Performs pattern-based replacements on the SMILES string.
    """
    if not smiles:
        return smiles
    
    # Define cleanup rules (pattern -> replacement)
    cleanup_rules = [
        ('C(=O[H])', 'C(=O)'),       # Remove H from carbonyl oxygen
        ('CO([H])[H]', 'CO[H]'),     # Remove extra H from hydroxyl
        ('CN([H])([H])[H]', 'CN([H])[H]'),  # Remove extra H from amine
        
        # Additional rules that might be helpful
        ('N(=O)[H]', 'N=O'),         # Remove H from N=O
        ('O=[NH]', 'O=N'),           # Remove H from O=N
        ('[H]O=', 'O='),             # Remove H from O= (common issue)
        ('=[OH]', '=O'),             # Fix =OH to =O
        
        # Fix common nitrogen patterns
        ('N([H])([H])([H])', 'N([H])[H]'),  # NH3 -> NH2
        ('=N([H])[H]', '=N[H]'),     # =NH2 -> =NH
        ('=N([H])([H])', '=N[H]'),   # Alternative =NH2 -> =NH
    ]
    
    # Apply each rule
    cleaned_smiles = smiles
    replacements_made = False
    
    for pattern, replacement in cleanup_rules:
        if pattern in cleaned_smiles:
            old_smiles = cleaned_smiles
            cleaned_smiles = cleaned_smiles.replace(pattern, replacement)
            if old_smiles != cleaned_smiles:
                replacements_made = True
    
    return cleaned_smiles, replacements_made

def make_reasonable_tautomers(tautomers):
    """
    Process all tautomers to make them chemically reasonable
    by applying cleanup rules, then sanitizing.
    """
    reasonable_tautomers = []
    
    st.write("Applying chemical reasonability rules to tautomers...")
    
    for i, taut in enumerate(tautomers):
        try:
            # Convert to SMILES for pattern-based cleanup
            try:
                original_smiles = Chem.MolToSmiles(taut)
                #st.write(f"Processing tautomer {i+1}: {original_smiles}")
                
                # Apply cleanup rules
                cleaned_smiles, changes_made = cleanup_tautomer_smiles(original_smiles)
                
                if changes_made:
                    st.write(f"✓ Applied cleanup rules: {original_smiles} → {cleaned_smiles}")
                else:
                    st.write(f"✓ No cleanup needed for tautomer {i+1}")
                
                # Parse the cleaned SMILES
                cleaned_mol = Chem.MolFromSmiles(cleaned_smiles)
                
                # If cleaning made it invalid, fall back to original
                if cleaned_mol is None:
                    st.warning(f"⚠ Cleanup resulted in invalid structure, using original")
                    reasonable_tautomers.append(taut)
                else:
                    # Add hydrogens to the cleaned molecule
                    cleaned_mol_h = Chem.AddHs(cleaned_mol)
                    reasonable_tautomers.append(cleaned_mol_h)
            except Exception as e:
                st.warning(f"⚠ Error in SMILES processing for tautomer {i+1}: {e}")
                reasonable_tautomers.append(taut)  # Fall back to original
        except Exception as e:
            st.warning(f"⚠ Error processing tautomer {i+1}: {e}")
            reasonable_tautomers.append(taut)  # Fall back to original
    
    st.write(f"Made {len(reasonable_tautomers)} tautomers chemically reasonable")
    
    return reasonable_tautomers

    

def display_ready_tautomers(tautomers):
    """
    Process tautomers to make them ready for display, prioritizing
    showing all tautomers even if energy calculation isn't possible.
    """
    result_tautomers = []
    energy_capable_tautomers = []
    
    #st.write(f"Processing {len(tautomers)} tautomers for display...")
    
    for i, taut in enumerate(tautomers):
        try:
            # Get SMILES for reporting
            try:
                taut_smiles = Chem.MolToSmiles(taut)
                #st.write(f"Processing tautomer {i+1}: {taut_smiles}")
            except:
                st.write(f"Processing tautomer {i+1}")
            
            # Simple sanitization for display
            display_taut = simple_sanitize_tautomer(taut)
            
            # Add to display list
            if display_taut is not None:
                result_tautomers.append(display_taut)
                st.write(f"✓ Tautomer {i+1} ready for display")
                
                # Check if we can calculate energy
                try:
                    mol_3d = Chem.Mol(display_taut)
                    AllChem.EmbedMolecule(mol_3d)
                    props = AllChem.MMFFGetMoleculeProperties(mol_3d)
                    if props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                            # This tautomer can have its energy calculated
                            energy_capable_tautomers.append(display_taut)
                            st.write(f"✓ Tautomer {i+1} also valid for energy calculation")
                except:
                    # Can't calculate energy, but still display it
                    st.write(f"⚠ Tautomer {i+1} can be displayed but not energy-calculated")
            else:
                st.warning(f"✗ Could not prepare tautomer {i+1} for display, skipping")
        except Exception as e:
            st.warning(f"Error processing tautomer {i+1}: {e}")
    
    st.write(f"Prepared {len(result_tautomers)}/{len(tautomers)} tautomers for display")
    st.write(f"Of these, {len(energy_capable_tautomers)}/{len(result_tautomers)} can have energies calculated")
    
    # Ensure we always have at least one molecule
    if not result_tautomers and tautomers:
        st.warning("All tautomer processing failed! Using original molecules")
        result_tautomers = tautomers
        
    return result_tautomers, energy_capable_tautomers
    
def calculate_mmff_energy_if_possible(mol):
    """
    Try to calculate MMFF energy, but return None without warnings if not possible.
    """
    if mol is None:
        return None
    
    try:
        # Create a copy for 3D
        mol_3d = Chem.Mol(mol)
        
        # Generate a 3D conformation
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        
        if result == -1:
            return None
        
        # Set up MMFF force field
        props = AllChem.MMFFGetMoleculeProperties(mol_3d)
        if props is None:
            return None
            
        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props)
        if ff is None:
            return None
        
        # Calculate energy
        energy = ff.CalcEnergy()
        return energy
    except:
        return None

def cleanup_with_smarts(mol):
    """
    Apply chemical cleanup rules using SMARTS patterns to fix valence issues.
    This is more precise than simple SMILES string replacements.
    """
    if mol is None:
        return mol
    
    try:
        # Work with a copy
        editable_mol = Chem.EditableMol(mol)
        
        # Define SMARTS patterns to match problematic atoms
        patterns = [
            # Nitrogen with 4 connections (3 hydrogens + carbon)
            "[N;X4;H3;v4]",  # Nitrogen with 4 connections, 3 hydrogens, valence 4
            
            # Oxygen with 3 connections (2 hydrogens + carbon)
            "[O;X3;H2;v3]",  # Oxygen with 3 connections, 2 hydrogens, valence 3
            
            # Excessively protonated hydroxyl
            "[O;X3;H2]-[C,c]",  # Oxygen with 2 H attached to carbon
            
            # Excessively protonated carbonyl
            "[O;X2;H1]=[C,c]",  # OH attached to C by double bond
        ]
        
        # For each pattern, find matches and fix them
        atom_indices_to_fix = set()
        
        for pattern in patterns:
            patt = Chem.MolFromSmarts(pattern)
            matches = mol.GetSubstructMatches(patt)
            
            for match in matches:
                # Add the first atom of each match (the problematic atom)
                atom_indices_to_fix.add(match[0])
        
        # Collect hydrogens to remove
        h_atoms_to_remove = []
        
        for atom_idx in atom_indices_to_fix:
            atom = mol.GetAtomWithIdx(atom_idx)
            element = atom.GetSymbol()
            
            # Find hydrogen atoms attached to this atom
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    h_atoms_to_remove.append(neighbor.GetIdx())
                    
                    # Limit the number of hydrogens to remove based on element
                    if element == 'N' and len(h_atoms_to_remove) >= 1:  # Remove at most 1 H from N
                        break
                    elif element == 'O' and len(h_atoms_to_remove) >= 1:  # Remove at most 1 H from O
                        break
        
        # Sort in reverse order to avoid changing indices
        h_atoms_to_remove.sort(reverse=True)
        
        # Remove the hydrogens
        for h_idx in h_atoms_to_remove:
            editable_mol.RemoveAtom(h_idx)
        
        # Get the modified molecule
        modified_mol = editable_mol.GetMol()
        
        return modified_mol
    except Exception as e:
        st.warning(f"Error in SMARTS cleanup: {e}")
        return mol

def make_reasonable_tautomers_enhanced(tautomers):
    """
    Process all tautomers to make them chemically reasonable
    using both SMILES pattern replacements and SMARTS-based cleanup.
    """
    reasonable_tautomers = []
    
    st.write("Applying enhanced chemical reasonability rules to tautomers...")
    
    for i, taut in enumerate(tautomers):
        try:
            # Get original SMILES for comparison
            try:
                original_smiles = Chem.MolToSmiles(taut)
                #st.write(f"Processing tautomer {i+1}: {original_smiles}")
            except:
                #st.write(f"Processing tautomer {i+1}")
                original_smiles = "unknown"
            
            # Apply SMILES-based cleanup first (simpler)
            try:
                # First try cleaning via SMILES pattern replacement
                cleaned_smiles, changes_made_smiles = cleanup_tautomer_smiles(original_smiles)
                
                if changes_made_smiles:
                    st.write(f"✓ Applied SMILES cleanup: {original_smiles} → {cleaned_smiles}")
                    # Parse the cleaned SMILES
                    intermediate_mol = Chem.MolFromSmiles(cleaned_smiles)
                    
                    # If cleaning made it invalid, fall back to original
                    if intermediate_mol is None:
                        st.warning(f"⚠ SMILES cleanup resulted in invalid structure, using original")
                        intermediate_mol = taut
                    else:
                        # Add hydrogens
                        intermediate_mol = Chem.AddHs(intermediate_mol)
                else:
                    # No SMILES changes needed
                    intermediate_mol = taut
            except Exception as e:
                st.warning(f"⚠ Error in SMILES processing: {e}")
                intermediate_mol = taut  # Fall back to original
            
            # Then apply more precise SMARTS-based cleanup
            try:
                # Apply SMARTS-based cleanup for more advanced fixes
                fixed_mol = cleanup_with_smarts(intermediate_mol)
                
                if fixed_mol:
                    # Check if structure changed
                    try:
                        final_smiles = Chem.MolToSmiles(fixed_mol)
                        intermediate_smiles = Chem.MolToSmiles(intermediate_mol)
                        
                        if final_smiles != intermediate_smiles:
                            st.write(f"✓ Applied SMARTS cleanup: {intermediate_smiles} → {final_smiles}")
                    except:
                        pass
                    
                    reasonable_tautomers.append(fixed_mol)
                else:
                    # If SMARTS cleanup failed, use the intermediate result
                    reasonable_tautomers.append(intermediate_mol)
            except Exception as e:
                st.warning(f"⚠ Error in SMARTS processing: {e}")
                # Fall back to intermediate result
                reasonable_tautomers.append(intermediate_mol)
        except Exception as e:
            st.warning(f"⚠ Error processing tautomer {i+1}: {e}")
            reasonable_tautomers.append(taut)  # Fall back to original
    
    st.write(f"Made {len(reasonable_tautomers)} tautomers chemically reasonable")
    
    return reasonable_tautomers

def advanced_process_tautomers_enhanced(tautomers):
    """
    Enhanced tautomer processing pipeline:
    1. Make tautomers display-ready
    2. Apply enhanced chemical reasonability rules (SMILES + SMARTS)
    3. Identify which can have energies calculated
    """
    # First make tautomers display-ready
    display_tautomers, _ = display_ready_tautomers(tautomers)
    
    if not display_tautomers:
        st.warning("No display-ready tautomers found!")
        return [], []
    
    # Apply enhanced chemical reasonability rules
    reasonable_tautomers = make_reasonable_tautomers_enhanced(display_tautomers)
    
    # Find which tautomers can have their energies calculated
    energy_capable = []
    for i, taut in enumerate(reasonable_tautomers):
        try:
            # Test if we can calculate energy
            mol_3d = Chem.Mol(taut)
            embedded = AllChem.EmbedMolecule(mol_3d) != -1
            
            if embedded:
                props = AllChem.MMFFGetMoleculeProperties(mol_3d)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props)
                    if ff is not None:
                        # Test calculation
                        energy = ff.CalcEnergy()
                        energy_capable.append(taut)
                        st.write(f"✓ Tautomer {i+1} can have energy calculated: {energy:.4f} kcal/mol")
        except:
            pass  # Skip without warning
    
    st.write(f"Found {len(energy_capable)}/{len(reasonable_tautomers)} tautomers capable of energy calculation")
    
    return reasonable_tautomers, energy_capable


# Streamlit UI
st.title("Tautomer Enumerator")
st.write("This app enumerates tautomers for a given molecule using RDKit's tautomer enumerator.")

smiles_input = st.text_input("Enter SMILES string:", "C1(=NN=C(C2=C1C=CC=C2)O)N")

# Advanced settings in sidebar
st.sidebar.title("Advanced Settings")
max_tautomers = st.sidebar.slider("Maximum number of tautomers to generate", 10, 1000, 100)
show_3d = st.sidebar.checkbox("Show 3D structures", value=False)
energy_calculation = st.sidebar.checkbox("Calculate MMFF energies", value=True)

# Example molecules with clean SMILES
st.write("""
### Example SMILES to try:
- Pyrazolone: `O=c1cc[nH][nH]1` 
- Diketone: `CC(=O)C(=O)CC`
- Azoxy compound: `c1ccc2c(c1)C(=NN=C2O)N`
- Acetaldehyde: `CC=O`
""")

# Final button handler with enhanced SMARTS-based cleanup rules
if st.button("Enumerate Tautomers"):
    if not smiles_input:
        st.error("Please enter a valid SMILES string")
    else:
        with st.spinner("Enumerating tautomers..."):
            # Enumerate tautomers
            tautomers, canonical_taut = enumerate_tautomers(smiles_input, max_tautomers)
            
            if tautomers is None:
                st.error(f"Error processing SMILES: {canonical_taut}")
            else:
                # Process tautomers using the enhanced pipeline
                tautomers, energy_capable_tautomers = advanced_process_tautomers_enhanced(tautomers)
                
                # Fix canonical tautomer with same approach
                try:
                    canonical_taut = Chem.MolFromSmiles(Chem.MolToSmiles(canonical_taut))
                    if canonical_taut:
                        canonical_taut = Chem.AddHs(canonical_taut)
                        # Apply SMARTS cleanup to canonical tautomer too
                        canonical_taut = cleanup_with_smarts(canonical_taut)
                except:
                    # If canonical tautomer processing fails, create from scratch
                    canonical_taut = Chem.MolFromSmiles(smiles_input)
                    if canonical_taut:
                        canonical_taut = Chem.AddHs(canonical_taut)
                
                st.success(f"Successfully prepared {len(tautomers)} chemically reasonable tautomers")
                
                # Display original molecule
                orig_mol = Chem.MolFromSmiles(smiles_input)
                if orig_mol is not None:
                    orig_mol = Chem.AddHs(orig_mol)
                    canonical_smiles = Chem.MolToSmiles(orig_mol)

                    st.subheader("Original Molecule")
                    st.write(f"Original SMILES: {smiles_input}")
                    st.write(f"Canonical SMILES: {canonical_smiles}")
                    
                    # Display as SVG
                    svg = display_molecule_svg(orig_mol, "Original Molecule", 400, 300)
                    st.components.v1.html(svg, height=350, width=450)
                
                # Prepare data for display
                taut_data = []
                
                # Calculate energies if requested
                with st.spinner("Calculating properties for each tautomer..."):
                    for i, taut in enumerate(tautomers):
                        try:
                            taut_smiles = Chem.MolToSmiles(taut)
                            
                            # Calculate energy if requested (only for compatible tautomers)
                            energy = None
                            if energy_calculation:
                                # Check if tautomer is in energy-capable list
                                for energy_taut in energy_capable_tautomers:
                                    if Chem.MolToSmiles(energy_taut) == taut_smiles:
                                        energy = calculate_mmff_energy_if_possible(energy_taut)
                                        break
                            
                            # Check if this is the canonical tautomer
                            is_canonical = False
                            try:
                                is_canonical = (canonical_taut and Chem.MolToSmiles(taut) == Chem.MolToSmiles(canonical_taut))
                            except:
                                pass
                            
                            taut_data.append({
                                "ID": i + 1,
                                "SMILES": taut_smiles,
                                "Energy (kcal/mol)": energy if energy is not None else None,
                                "Is Canonical": "Yes" if is_canonical else "No",
                                "Molecule": taut
                            })
                        except Exception as e:
                            st.warning(f"Error processing tautomer {i+1}: {e}")
                
                # Sort by energy if available
                if energy_calculation:
                    # Filter out None energies and sort
                    taut_data_with_energy = [t for t in taut_data if t["Energy (kcal/mol)"] is not None]
                    taut_data_without_energy = [t for t in taut_data if t["Energy (kcal/mol)"] is None]
                    
                    taut_data_with_energy.sort(key=lambda x: x["Energy (kcal/mol)"])
                    taut_data = taut_data_with_energy + taut_data_without_energy
                
                # Create a DataFrame for display (without the molecule column)
                display_data = [{k: v for k, v in item.items() if k != "Molecule"} for item in taut_data]
                st.subheader("All Tautomers")
                st.dataframe(pd.DataFrame(display_data))
                
                # Draw tautomer structures
                st.subheader("Tautomer Structures")
                
                # Display tautomers in a grid with two columns
                num_tautomers = len(tautomers)
                for i in range(0, num_tautomers, 2):
                    cols = st.columns(2)
                    
                    for j in range(2):
                        if i + j < num_tautomers:
                            taut = tautomers[i + j]
                            with cols[j]:
                                try:
                                    smiles = Chem.MolToSmiles(taut)
                                    st.write(f"**Tautomer {i + j + 1}**")
                                    st.write(f"SMILES: `{smiles}`")
                                    
                                    # Only calculate energy for energy-capable tautomers
                                    energy = None
                                    if energy_calculation:
                                        for energy_taut in energy_capable_tautomers:
                                            if Chem.MolToSmiles(energy_taut) == smiles:
                                                energy = calculate_mmff_energy_if_possible(energy_taut)
                                                break
                                    
                                    if energy is not None:
                                        st.write(f"Energy: `{energy:.4f} kcal/mol`")
                                    else:
                                        st.write("Energy: Not calculated")
                                    
                                    # Display molecule
                                    img = Draw.MolToImage(taut, size=(250, 200))
                                    st.image(img)
                                except Exception as e:
                                    st.warning(f"Error displaying tautomer {i+j+1}: {e}")
                
                # 3D structure display
                if show_3d and len(tautomers) > 0:
                    st.markdown("---")
                    st.subheader("3D Structure Visualization")
                    
                    # Let user select which tautomer to view in 3D
                    selected_taut_index = st.selectbox(
                        "Choose a tautomer", 
                        range(len(tautomers)), 
                        format_func=lambda x: f"Tautomer {x+1}"
                    )
                    
                    # Display the selected tautomer in 3D
                    try:
                        selected_taut = tautomers[selected_taut_index]
                        st.write(f"Displaying 3D structure for Tautomer {selected_taut_index + 1}: {Chem.MolToSmiles(selected_taut)}")
                        
                        # Check if this tautomer is energy-capable for better 3D
                        is_energy_capable = False
                        for energy_taut in energy_capable_tautomers:
                            if Chem.MolToSmiles(energy_taut) == Chem.MolToSmiles(selected_taut):
                                is_energy_capable = True
                                selected_taut = energy_taut  # Use the energy-capable version
                                break
                        
                        # Generate 3D coordinates with appropriate method
                        if is_energy_capable:
                            mol_with_h = generate_3d_for_mol_reliable(selected_taut)
                        else:
                            # For non-energy capable, just do basic embedding
                            mol_with_h = Chem.Mol(selected_taut)
                            AllChem.Compute2DCoords(mol_with_h)
                        
                        # Convert to MolBlock for 3D display
                        mol_block = Chem.MolToMolBlock(mol_with_h)
                        
                        # Create 3D visualization
                        html_content = f"""
                        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
                        <div id="mol3d" style="height: 400px; width: 100%; position: relative;"></div>
                        <script>
                        (function() {{
                            let viewer = $3Dmol.createViewer(document.getElementById('mol3d'), {{backgroundColor: 'white'}});
                            let molData = `{mol_block}`;
                            viewer.addModel(molData, 'mol');
                            viewer.setStyle({{}}, {{stick: {{}}}});
                            viewer.zoomTo();
                            viewer.rotate(30, 'y');
                            viewer.render();
                        }})();
                        </script>
                        """
                        
                        st.components.v1.html(html_content, height=450)
                    except Exception as e:
                        st.error(f"Error in 3D visualization: {e}")
                
                # Energy comparison 
                if energy_calculation and energy_capable_tautomers:
                    valid_energies = []
                    for i, taut in enumerate(tautomers):
                        # Only include tautomers that can have energy calculated
                        taut_smiles = Chem.MolToSmiles(taut)
                        for energy_taut in energy_capable_tautomers:
                            if Chem.MolToSmiles(energy_taut) == taut_smiles:
                                energy = calculate_mmff_energy_if_possible(energy_taut)
                                if energy is not None:
                                    valid_energies.append((i, taut, energy))
                                break
                    
                    if valid_energies:
                        st.markdown("---")
                        st.subheader("Energy Comparison")
                        
                        # Sort by energy
                        valid_energies.sort(key=lambda x: x[2])
                        
                        # Create energy data for display
                        energy_data = pd.DataFrame([
                            {"Tautomer": f"Tautomer {i+1}", 
                             "SMILES": Chem.MolToSmiles(taut),
                             "Energy (kcal/mol)": f"{energy:.4f}",
                             "Relative Energy (kcal/mol)": f"{energy - valid_energies[0][2]:.4f}"
                            } 
                            for i, taut, energy in valid_energies
                        ])
                        
                        st.dataframe(energy_data)
                        
                        # Create chart
                        if len(valid_energies) > 1:
                            chart_data = pd.DataFrame({
                                "Tautomer": [f"Taut {i+1}" for i, _, _ in valid_energies],
                                "Energy (kcal/mol)": [e for _, _, e in valid_energies]
                            })
                            st.bar_chart(chart_data.set_index("Tautomer"))