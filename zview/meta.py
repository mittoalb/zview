#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata Viewer Component
Handles display and exploration of Zarr metadata
"""

import json
from PyQt5 import QtWidgets, QtCore


class ZarrMetadataExtractor:
    """Extract metadata from Zarr files (z5py compatible)"""
    
    @staticmethod
    def extract_metadata(zarr_group):
        """Extract metadata from Zarr group and arrays"""
        metadata = []
        
        if hasattr(zarr_group, 'attrs'):
            try:
                attrs_dict = dict(zarr_group.attrs)
                for key, value in attrs_dict.items():
                    value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "..."
                    metadata.append((f"/.zattrs/{key}", value_str, type(value).__name__))
            except:
                pass
        
        def visit_items(name, obj):
            if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
                if hasattr(obj, 'attrs'):
                    try:
                        attrs_dict = dict(obj.attrs)
                        for key, value in attrs_dict.items():
                            value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                            if len(value_str) > 500:
                                value_str = value_str[:500] + "..."
                            metadata.append((f"/{name}/.zattrs/{key}", value_str, type(value).__name__))
                    except:
                        pass
                
                metadata.append((f"/{name}/shape", str(obj.shape), "tuple"))
                metadata.append((f"/{name}/dtype", str(obj.dtype), "dtype"))
                if hasattr(obj, 'chunks'):
                    metadata.append((f"/{name}/chunks", str(obj.chunks), "tuple"))
                if hasattr(obj, 'compression'):
                    metadata.append((f"/{name}/compression", str(obj.compression), "str"))
        
        try:
            zarr_group.visititems(visit_items)
        except AttributeError:
            for key in zarr_group.keys():
                try:
                    obj = zarr_group[key]
                    visit_items(key, obj)
                except:
                    pass
        
        return metadata
    
    @staticmethod
    def extract_tree_structure(zarr_group):
        """Extract tree structure of Zarr hierarchy"""
        structure = []
        
        def visit_items(name, obj):
            if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
                structure.append((name, 'Dataset', obj.shape, obj.dtype))
            elif hasattr(obj, 'keys'):
                structure.append((name, 'Group', None, None))
        
        try:
            zarr_group.visititems(visit_items)
        except AttributeError:
            for key in zarr_group.keys():
                try:
                    obj = zarr_group[key]
                    visit_items(key, obj)
                except:
                    pass
        
        return structure


class MetadataViewer(QtWidgets.QWidget):
    """Widget for displaying Zarr metadata"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Attributes tab
        metadata_widget = QtWidgets.QWidget()
        metadata_layout = QtWidgets.QVBoxLayout(metadata_widget)
        
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self.filter_input = QtWidgets.QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter...")
        self.filter_input.textChanged.connect(self._filter_metadata)
        filter_layout.addWidget(self.filter_input)
        metadata_layout.addLayout(filter_layout)
        
        self.metadata_table = QtWidgets.QTableWidget()
        self.metadata_table.setColumnCount(3)
        self.metadata_table.setHorizontalHeaderLabels(['Path/Attribute', 'Value', 'Type'])
        self.metadata_table.horizontalHeader().setStretchLastSection(False)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        self.metadata_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.metadata_table.setAlternatingRowColors(True)
        self.metadata_table.setSortingEnabled(True)
        metadata_layout.addWidget(self.metadata_table)
        
        self.tab_widget.addTab(metadata_widget, "Attributes")

        # Structure tab
        structure_widget = QtWidgets.QWidget()
        structure_layout = QtWidgets.QVBoxLayout(structure_widget)
        
        self.structure_tree = QtWidgets.QTreeWidget()
        self.structure_tree.setHeaderLabels(['Path', 'Type', 'Shape', 'Dtype'])
        self.structure_tree.setAlternatingRowColors(True)
        structure_layout.addWidget(self.structure_tree)
        
        self.tab_widget.addTab(structure_widget, "Structure")
        
        layout.addWidget(self.tab_widget)
        
        self.status_label = QtWidgets.QLabel("No metadata loaded")
        self.status_label.setStyleSheet("color: #999; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def load_metadata(self, zarr_group):
        """Load metadata from Zarr group"""
        try:
            metadata = ZarrMetadataExtractor.extract_metadata(zarr_group)
            self._all_metadata = metadata
            self._populate_metadata_table(metadata)
            
            structure = ZarrMetadataExtractor.extract_tree_structure(zarr_group)
            self._populate_structure_tree(structure)
            
            self.status_label.setText(f"Loaded {len(metadata)} attributes from Zarr store")
            self.status_label.setStyleSheet("color: #4a4; padding: 5px;")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f44; padding: 5px;")
    
    def _populate_metadata_table(self, metadata):
        self.metadata_table.setSortingEnabled(False)
        self.metadata_table.setRowCount(len(metadata))
        
        for row, (path, value, dtype) in enumerate(metadata):
            path_item = QtWidgets.QTableWidgetItem(path)
            path_item.setFlags(path_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.metadata_table.setItem(row, 0, path_item)
            
            value_item = QtWidgets.QTableWidgetItem(str(value))
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item.setToolTip(str(value))
            self.metadata_table.setItem(row, 1, value_item)
            
            type_item = QtWidgets.QTableWidgetItem(dtype)
            type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.metadata_table.setItem(row, 2, type_item)
        
        self.metadata_table.setSortingEnabled(True)
        self.metadata_table.resizeColumnsToContents()
    
    def _populate_structure_tree(self, structure):
        self.structure_tree.clear()
        root = QtWidgets.QTreeWidgetItem(self.structure_tree)
        root.setText(0, '/')
        root.setText(1, 'Group')
        root.setExpanded(True)
        
        for path, obj_type, shape, dtype in sorted(structure):
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, path)
            item.setText(1, obj_type)
            if shape is not None:
                item.setText(2, str(shape))
            if dtype is not None:
                item.setText(3, str(dtype))
            root.addChild(item)
        
        self.structure_tree.expandAll()
        self.structure_tree.resizeColumnToContents(0)
    
    def _filter_metadata(self, text):
        if not hasattr(self, '_all_metadata'):
            return
        if not text:
            self._populate_metadata_table(self._all_metadata)
        else:
            filtered = [item for item in self._all_metadata if text.lower() in item[0].lower()]
            self._populate_metadata_table(filtered)
    
    def clear(self):
        self.metadata_table.setRowCount(0)
        self.structure_tree.clear()
        self.status_label.setText("No metadata loaded")
        self.status_label.setStyleSheet("color: #999; padding: 5px;")