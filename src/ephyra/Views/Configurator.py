#!/usr/bin/env python
# coding=utf-8
"""
 * This file is part of the Aietes Framework (https://github.com/andrewbolster/aietes)
 *
 * (C) Copyright 2013 Andrew Bolster (http://andrewbolster.info/) and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import logging
from wx.lib.agw.customtreectrl import CustomTreeCtrl
from wx.lib.intctrl import IntCtrl
from wx.lib.agw.floatspin import FloatSpin

from ephyra import wx
from datetime import datetime as dt
from aietes.Tools import generate_names, itersubclasses
from aietes.Behaviour import Behaviour


def timestamp():
    """


    :return:
    """
    return dt.now().strftime('%Y-%m-%d-%H-%M-%S.aietes')


class Configurator(wx.Panel):
    """
    The Configurator panel allows the user to generate aietes-compatible configurations

    The general operation is two fold;
        *Editing simulation environment
        *Editing simulation defaults for Nodes
        *Editing simulation defaults for Behaviours
        *Editing simulation defaults for Applications
        *Individually editing modifications to Nodes, Applications, and Behaviours

    The editable Characteristics of Nodes in this are:
        *Application selection
        *Behaviour Selection
        *Speeds (Cruising, Max)
        *Max Turn rate
        *Initial Position

    The editible Characteristics of Behaviours in this are:
        *Factors (Clumping, Repulsion, Schooling, Waypointing)
        *Distances (Collision Avoidance, Min/Max Neighbourhood)
        *Nearest Neighbour Count
        *Update rate

    The editable Characteristics of Applications in this are:
        *NONE #ToDo

    """
    initial_fleets = 1
    initial_nodes = 8

    sim_default = {
        "Config": {"Duration": 1000,
                   "Interval": 1,
                   "Name": timestamp}
    }
    fleet_default = {
        "Config": {}
    }
    node_default = {
        "Config": {"Max Speed": 2.3,
                   "Cruising Speed": 1.4,
                   "Max Turn Rate": 4.5,
                   "Initial Position": {"X": 0.0, "Y": 1.0, "Z": 2.0}
                   }
    }
    behaviour_default = {
        "Config": {"Protocol": [cls.__name__ for cls in itersubclasses(Behaviour)],
                   "Nearest Neighbours": 4,
                   "Max Neighbourhood": 100,
                   "Min Neighbourhood": 10,
                   "Clumping Factor": 0.125,
                   "Schooling Factor": 0.01,
                   "Repulsive Distance": 20,
                   "Repulsive Factor": 0.01,
                   "Waypoint Factor*": 0.01,
                   "Update Rate": 0.3
                   }
    }

    def __init__(self, parent, frame, *args, **kw):
        wx.Panel.__init__(self, parent, *args, **kw)
        self.window = wx.SplitterWindow(self, style=wx.SP_3D | wx.SP_BORDER)
        self.config_panel = wx.Panel(self.window)

        # Configure Tree Contents before generation
        self.config_tree = ListNode("Simulation", data=self.sim_default)
        self.fleets = ListNode("Fleets")
        self.defaults = ListNode("Defaults", [
            ListNode("Fleet", data=self.fleet_default),
            ListNode("Node", data=self.node_default),
            ListNode("Behaviour", data=self.behaviour_default)
        ])
        for i in range(self.initial_fleets):
            self.add_fleet(index=i, nodes=self.initial_nodes)

        self.config_tree.append(self.fleets)
        self.config_tree.append(self.defaults)

        self.tree = CustomTreeCtrl(self.window)
        self.root = root = self.tree.AddRoot(self.config_tree.get_label(),
                                             data=self.config_tree.get_data())
        self.build_tree(self.config_tree, self.root)

        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.on_click, self.tree)
        self.tree.ExpandAll()

        self.sizer = sizer = wx.BoxSizer(wx.VERTICAL)
        self.window.SplitVertically(self.tree, self.config_panel)

        sizer.Add(self.window, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        sizer.Fit(parent)
        self.set_config_panel(self.root)
        self.Layout()

    def on_click(self, evt):
        self.set_config_panel(evt.GetItem())

    def print_tree(self, item=None, prefix=None):
        root = item if item is not None else self.config_tree
        prefix = prefix if prefix is not None else ""
        for item in root:
            try:
                print "%s:" % str([prefix, item.get_label()])
                if len(item):
                    self.print_tree(item, prefix=prefix + "+")
            except AttributeError:
                print "%s" % item
            except TypeError:
                print "%s" % item

    def build_tree(self, config, tree_node):
        for config_node in config:
            new_node = self.tree.AppendItem(
                tree_node, config_node.get_label(), data=config_node.get_data())
            if not config_node.is_leaf():
                self.build_tree(config_node, new_node)

    def update_tree(self, item, recurse=None):
        node = None
        try:
            node = self.tree.GetPyData(item)
        except Exception as e:
            print("%s:%s" % (item.get_label(), e))

        child, boza = self.tree.GetFirstChild(item)
        for s in node:
            if child is not None and child.IsOk():
                ni = child
                child, boza = self.tree.GetNextChild(item, boza)
                self.tree.SetItemText(ni, s.get_label())
                self.tree.SetPyData(ni, s)
                # for wx 2.2.1
                # self.tree.SetItemData( ni, wxTreeItemData( s ) )
                self.tree.SetItemHasChildren(ni, len(s))
                if len(s) and recurse:
                    self.update_tree(ni, recurse)
            else:
                ni = self.tree.AppendItem(item, s.get_label())
                self.tree.SetPyData(ni, s)
                self.tree.SetItemHasChildren(ni, len(s))
        if child is not None and child.IsOk():
            # prev list was longer
            extra = []
            while child.IsOk():
                extra.append(child)
                child, boza = self.tree.GetNextChild(item, boza)
            map(self.tree.Delete, extra)
        else:
            raise RuntimeError, "Child is probably bad: %s:%s" % (node, child)

    def on_resize(self, event):
        """

        :param event:
        """
        self.Layout()

    @staticmethod
    def on_idle(event):
        """

        :param event:
        """
        pass

    def add_fleet(self, index, *args, **kw):
        """
        Add a fleet to the simulation
        :param index:
        :param args:
        :param kw:
        """

        fleetid = self.fleets.append(ListNode("%s" % kw.get("name", "Fleet %d" % index), [
            ListNode("Nodes"),
            ListNode("Behaviours", data=kw.get(
                "behaviours", self.defaults[2].get_data()))
        ])
                                     )
        for i in range(kw.get("nodes", 1)):
            self.add_node(fleetid)

    def add_node(self, fleetid, *args, **kw):
        """

        :param fleetid:
        :param args:
        :param kw:
        """
        node_names = [n.get_label() for n in self.fleets[fleetid][0]]
        myname = kw.get(
            "name", str(generate_names(1, existing_names=node_names)[0]))
        logging.info("Added %s to fleet %d" % (myname, fleetid))
        self.fleets[fleetid][0].append(
            ListNode("%s" % myname, data=self.defaults[1].get_data()))

    def set_config_panel(self, item):
        """
        Create and layout the widgets in the dialog
        :param item:
        """
        logging.info("Setting config for :%s" % item)
        data = None
        try:
            data = self.tree.GetPyData(item)
        except Exception as e:
            print("%s:%s" % (item.get_label(), e))

        logging.info("Clicked on %s" % data)
        self.current_selection = item

        main_sizer = self.config_panel.GetSizer()
        if main_sizer:
            widgets = self.config_panel.GetChildren()
            for widget in widgets:
                logging.info("Destroying: %s" % (str(widget)))
                widget.Destroy()
                self.Layout()
            logging.info("Removing: MainSizer")
            self.sizer.Remove(main_sizer)

        btn_sizer = wx.StdDialogButtonSizer()

        if data is not None:
            logging.info("Item has data")

            values = data["Config"]
            gridsizer = wx.FlexGridSizer(rows=len(values), cols=2)
            colsizer = wx.BoxSizer(wx.HORIZONTAL)

            self.widgetNames = values
            font = wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD)

            for key, value in values.iteritems():
                logging.info("Parsing: %s %s" % (key, value))

                lbl = wx.StaticText(self.config_panel, label=key)
                lbl.SetFont(font)

                # Deal with funky functions
                if hasattr(value, '__call__'):
                    value = value()

                # LIST VALUES
                if isinstance(value, list):
                    default = value[0]
                    choices = value[1:]
                    input_choice = wx.ComboBox(self.config_panel, value=default,
                                               choices=choices,
                                               style=wx.CB_READONLY,
                                               name=key)
                # STRING VALUES
                elif isinstance(value, (basestring, unicode)):
                    input_choice = wx.TextCtrl(
                        self.config_panel, value=value, name=key)
                # INTEGER VALUES
                elif isinstance(value, int):
                    input_choice = IntCtrl(self.config_panel, value=value, name=key)
                # FLOAT VALUES
                elif isinstance(value, float):
                    input_choice = FloatSpin(
                        self.config_panel, increment=0.01, value=value, name=key)
                    input_choice.SetFormat("%f")
                    input_choice.SetDigits(2)
                # DICT VALUES - Assume position or vector
                elif isinstance(value, dict):
                    input_choice = wx.FlexGridSizer(rows=len(value), cols=2)
                    for k, v in sorted(value.iteritems()):
                        i_lbl = wx.StaticText(self.config_panel, label=k)
                        i_lbl.SetFont(font)

                        widget = FloatSpin(
                            self.config_panel, increment=0.01, value=v, name=k)
                        widget.SetFormat("%f")
                        widget.SetDigits(2)
                        input_choice.AddMany(
                            [(thing, 0, wx.ALL | wx.EXPAND | wx.ALIGN_RIGHT, 5) for thing in (i_lbl, widget)])
                else:
                    raise NotImplementedError, "Value (%s, %s) has not been coped with by set_config_panel" % (
                        str(value),
                        type(value)
                    )
                gridsizer.AddMany(
                    [(thing, 0, wx.ALL | wx.ALIGN_RIGHT, 5) for thing in (lbl, input_choice)])

            colsizer.Add(gridsizer, 1, wx.EXPAND)

            save_btn = wx.Button(self.config_panel, wx.ID_OK, label="Save")
            save_btn.Bind(wx.EVT_BUTTON, self.on_save)
            btn_sizer.AddButton(save_btn)

            update_btn = wx.Button(self.config_panel, wx.ID_ANY, label="Update")
            update_btn.Bind(wx.EVT_BUTTON, self.on_update)
            btn_sizer.AddButton(update_btn)

            cancel_btn = wx.Button(self.config_panel, wx.ID_CANCEL)
            btn_sizer.AddButton(cancel_btn)
            btn_sizer.Realize()

            main_sizer = wx.BoxSizer(wx.VERTICAL)
            main_sizer.Add(colsizer, 0, wx.EXPAND | wx.ALL | wx.ALIGN_RIGHT)
            main_sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 5)
            self.config_panel.SetSizer(main_sizer)
        else:
            logging.info("Item has no data")

        self.Layout()

    @staticmethod
    def on_save(evt):
        """

        :param evt:
        """
        print(evt)

    def on_update(self, evt):
        """
        Save the current values for each object
        :param evt:
        """
        print(evt)
        for name in self.widgetNames:
            try:
                widget = wx.FindWindowByName(name)
                if isinstance(widget, wx.ComboBox):
                    selection = widget.GetValue()
                    choices = widget.GetItems()
                    choices.insert(0, selection)
                    value = choices
                else:
                    value = widget.GetValue()

                data = self.tree.GetPyData(self.current_selection)
                data['Config'][name] = value
                self.tree.SetPyData(self.current_selection, data)
            except Exception as E:
                logging.error("%s: %s" % (E, name))
                raise E


##########################
# Configurator Helpers
##########################


class ListNode(object):
    """

    :param title:
    :param children:
    :param data:
    :raise RuntimeError:
    """

    def __init__(self, title, children=None, data=None):
        self._nl = []
        if children is not None:
            if isinstance(children, list):
                self._nl = children
            elif isinstance(children, ListNode):
                self._nl = [children]
            else:
                raise RuntimeError, "Unknown Children:%s" % children
        self._tt = title
        self._data = data

    def __len__(self):
        return len(self._nl)

    def __getitem__(self, item):
        return self._nl[item]

    def __setitem__(self, key, value):
        self._nl[key] = value

    def append(self, item):
        """

        :param item:
        :return:
        """
        print item.get_label()
        self._nl.append(item)
        return len(self) - 1

    def get_label(self):
        """


        :return:
        """
        return self._tt

    def get_data(self):
        """


        :return:
        """
        return self._data

    def is_leaf(self):
        """


        :return:
        """
        return len(self) == 0
